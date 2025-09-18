import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from draem_model import ReconstructiveSubNetwork
from train_rd4ad import get_embeddings, anomaly_map, MVTecDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# 1. 載入 Student checkpoint
# --------------------------
ckpt_path = "./checkpoints/student_best_epoch50_pixelAUROC0.9876.pth"
ckpt = torch.load(ckpt_path, map_location=device)

student_model = ReconstructiveSubNetwork(in_channels=3, out_channels=3).to(device)
student_model.load_state_dict(ckpt["student_state_dict"])
student_model.eval()

# Teacher model (DRAEM)
teacher_ckpt = torch.load("draem_trained.pckl", map_location=device)
teacher_model = ReconstructiveSubNetwork(in_channels=3, out_channels=3).to(device)
teacher_model.load_state_dict(teacher_ckpt['reconstructive'])
teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False

# --------------------------
# 2. 準備驗證資料
# --------------------------
val_dataset = MVTecDataset(root="./mvtec_ad", category="bottle", split="test", resize=256)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# --------------------------
# 3. 計算 Pixel-level / Image-level AUROC
# --------------------------
all_pixel_scores, all_pixel_labels = [], []
all_img_scores, all_img_labels = [], []

with torch.no_grad():
    for imgs, (img_labels, pixel_masks) in val_loader:
        imgs, pixel_masks = imgs.to(device), pixel_masks.to(device)
        anomaly = anomaly_map(imgs)

        # Pixel-level
        all_pixel_scores.append(anomaly.cpu().numpy().ravel())
        all_pixel_labels.append(pixel_masks.cpu().numpy().ravel())

        # Image-level (max score)
        img_scores = anomaly.view(anomaly.size(0), -1).max(dim=1)[0]
        all_img_scores.append(img_scores.cpu().numpy())
        all_img_labels.append(img_labels.numpy())

# 合併所有 batch
all_pixel_scores = np.concatenate(all_pixel_scores)
all_pixel_labels = np.concatenate(all_pixel_labels)
all_img_scores = np.concatenate(all_img_scores)
all_img_labels = np.concatenate(all_img_labels)

# 計算 AUROC
pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
img_auroc = roc_auc_score(all_img_labels, all_img_scores)

# --------------------------
# 4. 輸出報表
# --------------------------
report_path = "./inference_results/rd4ad_evaluation_report.txt"
with open(report_path, "w") as f:
    f.write(f"Student Model Checkpoint: {ckpt_path}\n")
    f.write(f"Pixel-level AUROC: {pixel_auroc:.4f}\n")
    f.write(f"Image-level AUROC: {img_auroc:.4f}\n")

print(f"Evaluation done. Report saved to {report_path}")
print(f"Pixel-level AUROC: {pixel_auroc:.4f}, Image-level AUROC: {img_auroc:.4f}")
