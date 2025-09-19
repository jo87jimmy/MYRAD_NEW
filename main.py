import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms as T
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import random  # 亂數控制
import argparse  # 命令列參數處理

from model_unet import ReconstructiveSubNetwork,StudentReconstructiveSubNetwork  # 你的 DRAEM 模型

def setup_seed(seed):
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保證結果可重現
    torch.backends.cudnn.benchmark = False  # 關閉自動最佳化搜尋

# =======================
# Dataset
# =======================
# 定義 MVTecDataset 類別，繼承自 PyTorch 的 Dataset，用於載入 MVTec 異常偵測資料集
class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category="bottle", split="train", resize=256):
        # 設定資料根目錄
        self.root = root
        # 指定類別（如 bottle、capsule 等）
        self.category = category
        # 指定資料切分方式（train 或 test）
        self.split = split
        # 設定影像資料夾路徑
        self.img_dir = os.path.join(root, category, split)
        # 設定 ground truth 遮罩資料夾路徑
        self.gt_dir = os.path.join(root, category, "ground_truth")
        # 初始化影像路徑、標籤與遮罩路徑的清單
        self.data, self.labels, self.masks = [], [], []

        # 遍歷影像資料夾中的所有子類別（如 good、broken_small 等）
        for defect_type in sorted(os.listdir(self.img_dir)):
            img_folder = os.path.join(self.img_dir, defect_type)
            # 若不是資料夾則跳過（排除非類別資料）
            if not os.path.isdir(img_folder):
                continue
            # 遍歷該類別資料夾中的所有影像檔案
            for f in sorted(os.listdir(img_folder)):
                # 若為正常類別（good），則不含遮罩
                img_path = os.path.join(img_folder, f)
                if defect_type == "good":
                    self.data.append(img_path)# 儲存影像路徑
                    self.labels.append(0) # 標記為正常樣本（label=0）
                    self.masks.append(None) # 遮罩為 None
                else:
                    # 若為異常類別，則根據影像檔名推導遮罩路徑
                    mask_path = os.path.join(self.gt_dir, defect_type, f.replace(".png","_mask.png"))
                    self.data.append(img_path) # 儲存影像路徑
                    self.labels.append(1) # 標記為異常樣本（label=1）
                    self.masks.append(mask_path) # 儲存遮罩路徑

        # 定義影像的轉換流程：調整大小並轉為 Tensor
        self.transform = T.Compose([
            T.Resize((resize, resize)),# 調整影像尺寸為指定大小
            T.ToTensor()# 轉換為 PyTorch Tensor 格式
        ])

        # 定義遮罩的轉換流程：使用最近鄰插值法調整大小並轉為 Tensor
        self.mask_transform = T.Compose([
            T.Resize((resize, resize), interpolation=Image.NEAREST),# 避免遮罩模糊
            T.ToTensor()# 轉換為 Tensor 格式
        ])

    # 回傳資料集的總長度（樣本數）
    def __len__(self):
        return len(self.data)
    # 定義資料集的索引存取方式
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("RGB")# 讀取指定索引的影像並轉為 RGB 格式
        img = self.transform(img)# 套用影像轉換流程（resize + tensor）
        label = self.labels[idx] # 取得該影像的標籤（0 或 1）
        mask_path = self.masks[idx]# 取得該影像對應的遮罩路徑（可能為 None）

        # 若遮罩為 None（正常樣本），則建立全黑遮罩
        if mask_path is None:
            mask = torch.zeros((1, img.shape[1], img.shape[2]))# 單通道全 0 遮罩
        else:
            mask = Image.open(mask_path).convert("L")# 否則載入遮罩圖並轉為灰階
            mask = self.mask_transform(mask)# 套用遮罩轉換流程（resize + tensor）
            mask = (mask>0.5).float()# 將遮罩二值化（大於 0.5 的視為異常區域）
         # 回傳影像、標籤與遮罩（標籤為 tensor，遮罩為 float tensor）
        return img, (torch.tensor(label, dtype=torch.long), mask)

# =======================
# Utilities
# =======================
# 定義函式 get_embeddings，用來從模型中提取中間層的特徵圖（feature maps）
def get_embeddings(model, x):
    # 初始化特徵儲存清單 feats，以及 hook 物件清單 hooks
    feats, hooks = [], []

    # 定義 forward hook 函式，當層執行 forward 時會將輸出結果加入 feats 清單
    def hook_fn(_, __, output):
        feats.append(output)

    # 遍歷模型中的所有模組（module）
    for layer in model.modules():
        # 若模組是卷積層（Conv2d），則註冊 forward hook
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn))

    # 執行模型 forward，觸發所有已註冊的 hook，並將特徵儲存到 feats
    _ = model(x)

    # 移除所有 hook，避免重複註冊或記憶體洩漏
    for h in hooks: h.remove()

    # 回傳所有提取到的特徵圖
    return feats

# 建立餘弦相似度計算器，指定比較維度為通道維度（dim=1）
cos = nn.CosineSimilarity(dim=1)
# 定義 RD4AD 的損失函數，用來比較教師與學生模型的特徵差異
def rd4ad_loss(teacher_feats, student_feats):
    # 初始化總損失為 0
    loss = 0
    # 同時遍歷教師與學生模型的特徵列表（每層對應一組）
    for t,s in zip(teacher_feats, student_feats):
        # 將教師特徵展平成 (batch_size, -1)，並在通道維度上進行 L2 正規化
        t = nn.functional.normalize(t.flatten(1), dim=1)
        # 將學生特徵展平成 (batch_size, -1)，並在通道維度上進行 L2 正規化
        s = nn.functional.normalize(s.flatten(1), dim=1)
        # 計算教師與學生特徵的餘弦相似度，並轉換為異常分數（1 - 相似度）
        # torch.mean 表示對整個 batch 取平均，作為該層的損失
        loss += torch.mean(1 - cos(t, s))
    # 回傳所有層損失的總和，作為最終損失值
    return loss

# 定義 anomaly_map 函式，用來產生異常分數圖（score map），比較教師與學生模型的特徵差異
def anomaly_map(imgs, teacher_model, student_model):
    # 停用梯度計算，加速推論並節省記憶體
    with torch.no_grad():
        # 從教師模型提取特徵圖（feature maps）
        t_feats = get_embeddings(teacher_model, imgs)
        # 從學生模型提取特徵圖
        s_feats = get_embeddings(student_model, imgs)
    # 初始化異常分數圖的儲存清單
    score_maps = []
    # 同時遍歷教師與學生模型的每層特徵圖
    for t,s in zip(t_feats, s_feats):
        # 對教師特徵圖進行 L2 正規化，沿著通道維度（dim=1）
        t = nn.functional.normalize(t, dim=1)
        # 對學生特徵圖進行 L2 正規化
        s = nn.functional.normalize(s, dim=1)
        # 計算教師與學生特徵的相似度（內積），並取平均後轉換為異常分數（1 - 相似度）
        # keepdim=True 保留通道維度，方便後續插值與拼接
        diff = 1 - torch.mean(t*s, dim=1, keepdim=True)
        # 將異常分數圖插值回原始影像大小，使用雙線性插值（bilinear）
        diff = nn.functional.interpolate(diff, size=imgs.shape[2:], mode="bilinear")
        # 將該層的異常分數圖加入清單
        score_maps.append(diff)
    # 將所有層的異常分數圖堆疊後取平均，得到最終的異常圖
    return torch.mean(torch.stack(score_maps), dim=0)

def anomaly_map_student_recon(imgs, student_model):
    with torch.no_grad():
        # Autoencoder 前向傳播：重建影像
        recon_imgs = student_model(imgs)

        # 計算重建誤差 (MSE)，當作 anomaly map
        anomaly = torch.mean((imgs - recon_imgs) ** 2, dim=1, keepdim=True)

        # anomaly map 形狀: (B, 1, H, W)
        return anomaly

# =======================
# Evaluation
# =======================
def evaluate(student_model, teacher_model, val_loader, device, writer=None, epoch=None):
    # 將學生模型設為推論模式，停用 Dropout、BatchNorm 等訓練專用機制
    student_model.eval()
    # 初始化儲存所有像素級分數與標籤的清單
    all_pixel_scores, all_pixel_labels = [], []
    # 初始化儲存所有影像級分數與標籤的清單
    all_img_scores, all_img_labels = [], []
    # 停用梯度計算，加速推論並節省記憶體
    with torch.no_grad():
        # 遍歷驗證資料集的每個批次
        for imgs, (img_labels, pixel_masks) in val_loader:
            # 將影像與像素遮罩移動到指定裝置（GPU 或 CPU）
            imgs, pixel_masks = imgs.to(device), pixel_masks.to(device)

            # 計算異常圖（anomaly map），比較教師與學生模型的特徵差異
            anomaly = anomaly_map(imgs, teacher_model, student_model)

            # 將異常圖轉為 NumPy 並展平，儲存像素級分數
            all_pixel_scores.append(anomaly.cpu().numpy().ravel())
            # 將像素遮罩轉為 NumPy 並展平，儲存像素級標籤
            all_pixel_labels.append(pixel_masks.cpu().numpy().ravel())

            # 將異常圖展平為 (batch_size, num_pixels)，取每張圖的最大異常分數作為影像級分數
            img_scores = anomaly.view(anomaly.size(0), -1).max(dim=1)[0]
            # 儲存影像級分數與標籤
            all_img_scores.append(img_scores.cpu().numpy())
            all_img_labels.append(img_labels.numpy())

        # 若有提供 TensorBoard writer 且指定 epoch，則進行前 4 張影像的可視化
        if writer is not None and epoch is not None:
            # 取出驗證資料集中一個批次的影像與標籤
            val_imgs_vis, (val_labels_vis, val_masks_vis) = next(iter(val_loader))
            # 將影像移動到指定裝置
            val_imgs_vis = val_imgs_vis.to(device)
            # 計算該批次的異常圖
            anomaly_vis = anomaly_map(val_imgs_vis, teacher_model, student_model)
            # 將異常圖正規化到 [0,1] 範圍，避免數值不穩定
            anomaly_norm = (anomaly_vis - anomaly_vis.min()) / (anomaly_vis.max() - anomaly_vis.min() + 1e-8)
            # 建立原始影像的網格圖（前 4 張）
            input_grid = make_grid(val_imgs_vis[:4], nrow=4)
            # 建立異常圖的網格圖（轉為 RGB）
            anomaly_grid = make_grid(anomaly_norm[:4].repeat(1,3,1,1), nrow=4)
            # 建立 Ground Truth 遮罩的網格圖（轉為 RGB）
            mask_grid = make_grid(val_masks_vis[:4].repeat(1,3,1,1), nrow=4)
            # 將三種圖像加入 TensorBoard，標記為當前 epoch
            writer.add_image("Val/Input", input_grid, epoch)
            writer.add_image("Val/AnomalyMap", anomaly_grid, epoch)
            writer.add_image("Val/GT_Mask", mask_grid, epoch)
    # 將所有批次的像素級分數與標籤合併為一個大陣列
    all_pixel_scores = np.concatenate(all_pixel_scores)
    all_pixel_labels = np.concatenate(all_pixel_labels)
    # 將所有批次的影像級分數與標籤合併為一個大陣列
    all_img_scores = np.concatenate(all_img_scores)
    all_img_labels = np.concatenate(all_img_labels)
    # 計算像素級 AUROC（Area Under ROC Curve）
    pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
    # 計算影像級 AUROC
    img_auroc = roc_auc_score(all_img_labels, all_img_scores)
    # 回傳兩種 AUROC 指標
    return pixel_auroc, img_auroc

# =======================
# Main Pipeline
# =======================
def main():
       # 解析命令列參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='bottle', type=str)  # 訓練類別
    parser.add_argument('--epochs', default=25, type=int)  # 訓練回合數
    parser.add_argument('--arch', default='wres50', type=str)  # 模型架構
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--train_bool', action='store_true', help='是否進行訓練')
    args = parser.parse_args()

    setup_seed(111)  # 固定隨機種子
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = f'./mvtec'  # 訓練資料路徑

    # Load datasets
    # 載入訓練資料集，指定根目錄、類別、資料切分方式為 "train"，並將影像尺寸調整為 256x256
    train_dataset = MVTecDataset(root=path, category=args.category, split="train", resize=256)
    # 建立訓練資料的 DataLoader，設定每批次大小為 16，打亂資料順序，使用 4 個執行緒加速載入
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)

    # 載入驗證資料集，切分方式為 "test"，同樣調整影像尺寸為 256x256
    val_dataset = MVTecDataset(root=path, category=args.category, split="test", resize=256)
    # 建立驗證資料的 DataLoader，設定每批次大小為 8，不打亂資料順序，使用 4 個執行緒加速載入
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    # Load teacher
    # 載入教師模型的檢查點（checkpoint）檔案，並指定載入到的裝置（如 GPU 或 CPU）
    teacher_ckpt = torch.load("DRAEM_seg_large_ae_large_0.0001_800_bs8_bottle_.pckl", map_location=device,weights_only=True)
    # 建立教師模型的結構，輸入與輸出通道皆為 3（RGB），並移動到指定裝置上
    teacher_model = ReconstructiveSubNetwork(in_channels=3, out_channels=3).to(device)
    # 將教師模型的參數載入至模型中，使用 checkpoint 中的 'reconstructive' 欄位
    teacher_model.load_state_dict(teacher_ckpt)
    # 將教師模型設為評估模式，停用 Dropout、BatchNorm 等訓練專用機制
    teacher_model.eval()
    # 將教師模型的所有參數設為不可訓練，避免在後續訓練中被更新
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Student model
    #dropout 防止過擬合，幫助學生模型泛化，避免過擬合教師模型提取的特徵。在蒸餾訓練時，讓學生模型學到更穩健的特徵，而不是完全模仿教師模型的單一路徑
    student_model = StudentReconstructiveSubNetwork(in_channels=3, out_channels=3,base_width=64,dropout_rate=0.2).to(device)
    #定義學生模型優化器和學習率排程器
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

    # 主儲存資料夾路徑
    save_root = "./save_files"

    # 若主資料夾不存在，則建立
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # TensorBoard
    # 建立 TensorBoard 的紀錄器，將訓練過程的指標與圖像輸出到指定目錄 "./save_files"
    writer = SummaryWriter(log_dir=save_root)
    # 指定模型檢查點（checkpoint）儲存的資料夾路徑
    # 模型檢查點儲存路徑
    checkpoint_dir = os.path.join(save_root, "checkpoints")
    # 如果檢查點資料夾不存在，則建立該資料夾（exist_ok=True 表示若已存在則不報錯）
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 初始化最佳像素級 AUROC（Area Under ROC Curve）指標為 0.0，用於追蹤模型效能
    best_pixel_auroc = 0.0
    # 從參數中取得訓練的總輪數（epochs）
    epochs = args.epochs
    # 初始化全域步數計數器，用於 TensorBoard 記錄
    global_step = 0
    torch.cuda.empty_cache()

    Training = args.train_bool

    if Training:
        # 開始進行多輪訓練迴圈
        for epoch in range(epochs):
            # 將學生模型設為訓練模式，啟用 Dropout、BatchNorm 等訓練機制
            student_model.train()
            # 遍歷訓練資料集的每個批次
            for imgs, _ in train_loader:
                # 將影像資料移動到指定裝置（GPU 或 CPU）
                imgs = imgs.to(device)
                # 使用教師模型提取特徵，並停用梯度計算以節省記憶體與加速推論
                with torch.no_grad():
                    teacher_feats = get_embeddings(teacher_model, imgs)
                # 使用學生模型提取特徵
                student_feats = get_embeddings(student_model, imgs)
                # 計算教師與學生特徵之間的差異損失（Loss）
                loss = rd4ad_loss(teacher_feats, student_feats)
                # 清除先前的梯度
                optimizer.zero_grad()
                # 反向傳播計算梯度
                loss.backward()
                # 更新學生模型的參數
                optimizer.step()
                # 將訓練損失記錄到 TensorBoard，標記為 "Train/Loss"
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                # 步數累加，用於追蹤訓練進度
                global_step += 1
            # 顯示目前 epoch 的訓練損失
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {loss.item():.4f}")

            # Validation
            # 執行驗證流程，計算像素級與影像級的 AUROC 指標
            pixel_auroc, img_auroc = evaluate(student_model, teacher_model, val_loader, device, writer, epoch)
            # 將驗證結果記錄到 TensorBoard
            writer.add_scalar("Val/Pixel_AUROC", pixel_auroc, epoch)
            writer.add_scalar("Val/Image_AUROC", img_auroc, epoch)
            # 顯示驗證結果
            print(f"Validation - Pixel AUROC: {pixel_auroc:.4f}, Image AUROC: {img_auroc:.4f}")

            # 若目前的像素級 AUROC 優於歷史最佳值，則儲存模型檢查點
            if pixel_auroc > best_pixel_auroc:
                # 更新最佳 AUROC 值
                best_pixel_auroc = pixel_auroc
                # 建立檢查點儲存路徑，包含 epoch 與 AUROC 指標
                ckpt_path = os.path.join(checkpoint_dir, f"student_best_epoch{epoch+1}_pixelAUROC{pixel_auroc:.4f}.pth")
                # 儲存模型狀態與優化器狀態到檔案
                torch.save({
                    "epoch": epoch+1,
                    "pixel_auroc": pixel_auroc,
                    "img_auroc": img_auroc,
                    "student_state_dict": student_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, ckpt_path)
                print(f"--> Saved best Student model to {ckpt_path}")
                # 建立Best模型的固定檔名
                ckpt_path_best = os.path.join(checkpoint_dir, f"student_best.pth")
                # 儲存模型狀態與優化器狀態到檔案
                torch.save({
                    "student_state_dict": student_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, ckpt_path_best)
                print(f"--> Saved best  model to {ckpt_path_best}")
        # 關閉 TensorBoard 紀錄器，釋放資源
        writer.close()
    torch.cuda.empty_cache()

    # --------------------------
    # Inference + visualization
    # --------------------------
    # 設定推論結果儲存的資料夾路徑為 save_root/inference_results
    inference_results = os.path.join(save_root, "inference_results")
    # 若資料夾不存在則建立，用來儲存推論圖像與報告
    os.makedirs(inference_results, exist_ok=True)

    if Training:
        # 載入最佳模型進行推論
        Best_model = StudentReconstructiveSubNetwork(
            in_channels=3,
            out_channels=3,
            base_width=64,      # 與訓練一致
            dropout_rate=0.2    # 與訓練一致
        ).to(device)
        # 載入 checkpoint
        ckpt_path = os.path.join(checkpoint_dir, "student_best.pth")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # 只載入模型權重
        Best_model.load_state_dict(checkpoint["student_state_dict"])

        # 設定為推論模式
        Best_model.eval()
        detection_model = Best_model  # 直接指向 Best_model
        print(f"Training done. Loaded best model from {ckpt_path} for inference.")
    else:
        detection_model = teacher_model.eval()
        print(f"No training. Using teacher model for inference.")
    # 停用梯度計算，加速推論並節省記憶體
    with torch.no_grad():
        # 遍歷驗證資料集的每個批次
        for i, (imgs, (labels, masks)) in enumerate(val_loader):
            # 將影像與遮罩資料移動到指定裝置（GPU 或 CPU）
            imgs, masks = imgs.to(device), masks.to(device)
            # 使用教師與學生模型計算異常圖（anomaly map）todo 因該是只有學生模型
            # anomaly = anomaly_map(imgs, teacher_model, detection_model)

            # Student (Autoencoder) 前向傳播：重建影像
            recon_imgs = anomaly_map_student_recon(imgs,detection_model)
            # 計算像素級異常分數 (重建誤差)
            anomaly = torch.mean((imgs - recon_imgs) ** 2, dim=1, keepdim=True)

            # 將異常圖正規化到 [0,1] 範圍，避免數值不穩定（加上 1e-8 防止除以 0）
            anomaly_norm = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-8)
            # 將單通道異常圖複製成 RGB 三通道格式，方便視覺化
            anomaly_rgb = anomaly_norm.repeat(1,3,1,1)

            # 將遮罩資料也複製成 RGB 三通道格式
            masks_rgb = masks.repeat(1,3,1,1)
            # 將原始影像、異常圖與遮罩圖沿著寬度方向拼接成一張比較圖
            combined = torch.cat([imgs, anomaly_rgb, masks_rgb], dim=3)
            # 儲存比較圖到指定路徑，檔名包含批次編號
            save_image(combined, f"{inference_results}/comparison_batch{i+1}.png")
            print(f"Saved batch {i+1} comparison to {inference_results}/comparison_batch{i+1}.png")

    # --------------------------
    # Full evaluation report
    # --------------------------
    # 執行模型評估，計算像素級與影像級的 AUROC 指標
    pixel_auroc, img_auroc = evaluate(detection_model, teacher_model, val_loader, device)
    # 設定報告儲存路徑為 inference_results/evaluation_report.txt
    report_path = os.path.join(inference_results, "evaluation_report.txt")
    # 將評估結果寫入報告檔案
    with open(report_path, "w") as f:
        f.write(f"Pixel-level AUROC: {pixel_auroc:.4f}\n")
        f.write(f"Image-level AUROC: {img_auroc:.4f}\n")
    # 顯示評估完成訊息與報告路徑
    print(f"Evaluation done. Report saved to {report_path}")
    print(f"Pixel-level AUROC: {pixel_auroc:.4f}, Image-level AUROC: {img_auroc:.4f}")

# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    main()
