import torch
from torchvision.utils import save_image, make_grid
from draem_model import ReconstructiveSubNetwork
from train_rd4ad import get_embeddings, anomaly_map, MVTecDataset, DataLoader

import os
os.makedirs("./inference_results", exist_ok=True)

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
# 3. 推理 + 儲存 anomaly map 並排 GT mask
# --------------------------
with torch.no_grad():
    for i, (imgs, (labels, masks)) in enumerate(val_loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        anomaly = anomaly_map(imgs)  # (B,1,H,W)

        # 正規化 anomaly map
        anomaly_norm = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-8)

        # repeat 3 channels
        anomaly_rgb = anomaly_norm.repeat(1,3,1,1)
        masks_rgb = masks.repeat(1,3,1,1)

        # 將原圖、anomaly map、GT mask 並排
        combined = torch.cat([imgs, anomaly_rgb, masks_rgb], dim=3)  # 在 width 上拼接
        save_image(combined, f"./inference_results/comparison_batch{i+1}.png")

        print(f"Saved batch {i+1} comparison to ./inference_results/comparison_batch{i+1}.png")
