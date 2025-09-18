import torch
from torchvision.utils import save_image, make_grid
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
# 3. 推理 + 儲存 anomaly map
# --------------------------
import os
os.makedirs("./inference_results", exist_ok=True)

with torch.no_grad():
    for i, (imgs, (labels, masks)) in enumerate(val_loader):
        imgs = imgs.to(device)
        anomaly = anomaly_map(imgs)  # (B,1,H,W)

        # 正規化 anomaly map
        anomaly_norm = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-8)

        # 合併原圖 + anomaly map 做 grid
        anomaly_grid = make_grid(anomaly_norm.repeat(1,3,1,1), nrow=4)
        input_grid = make_grid(imgs, nrow=4)

        save_image(input_grid, f"./inference_results/input_batch{i+1}.png")
        save_image(anomaly_grid, f"./inference_results/anomaly_batch{i+1}.png")

        print(f"Saved batch {i+1} results to ./inference_results/")
