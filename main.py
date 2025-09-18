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

from model_unet import ReconstructiveSubNetwork  # 你的 DRAEM 模型

# =======================
# Dataset
# =======================
class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category="bottle", split="train", resize=256):
        self.root = root
        self.category = category
        self.split = split
        self.img_dir = os.path.join(root, category, split)
        self.gt_dir = os.path.join(root, category, "ground_truth")

        self.data, self.labels, self.masks = [], [], []

        for defect_type in sorted(os.listdir(self.img_dir)):
            img_folder = os.path.join(self.img_dir, defect_type)
            if not os.path.isdir(img_folder):
                continue
            for f in sorted(os.listdir(img_folder)):
                img_path = os.path.join(img_folder, f)
                if defect_type == "normal":
                    self.data.append(img_path)
                    self.labels.append(0)
                    self.masks.append(None)
                else:
                    mask_path = os.path.join(self.gt_dir, defect_type, f.replace(".png","_mask.png"))
                    self.data.append(img_path)
                    self.labels.append(1)
                    self.masks.append(mask_path)

        self.transform = T.Compose([
            T.Resize((resize, resize)),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize((resize, resize), interpolation=Image.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        mask_path = self.masks[idx]

        if mask_path is None:
            mask = torch.zeros((1, img.shape[1], img.shape[2]))
        else:
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask>0.5).float()

        return img, (torch.tensor(label, dtype=torch.long), mask)

# =======================
# Utilities
# =======================
def get_embeddings(model, x):
    feats, hooks = [], []
    def hook_fn(_, __, output):
        feats.append(output)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn))
    _ = model(x)
    for h in hooks: h.remove()
    return feats

cos = nn.CosineSimilarity(dim=1)
def rd4ad_loss(teacher_feats, student_feats):
    loss = 0
    for t,s in zip(teacher_feats, student_feats):
        t = nn.functional.normalize(t.flatten(1), dim=1)
        s = nn.functional.normalize(s.flatten(1), dim=1)
        loss += torch.mean(1 - cos(t, s))
    return loss

def anomaly_map(imgs, teacher_model, student_model):
    with torch.no_grad():
        t_feats = get_embeddings(teacher_model, imgs)
        s_feats = get_embeddings(student_model, imgs)
    score_maps = []
    for t,s in zip(t_feats, s_feats):
        t = nn.functional.normalize(t, dim=1)
        s = nn.functional.normalize(s, dim=1)
        diff = 1 - torch.mean(t*s, dim=1, keepdim=True)
        diff = nn.functional.interpolate(diff, size=imgs.shape[2:], mode="bilinear")
        score_maps.append(diff)
    return torch.mean(torch.stack(score_maps), dim=0)

# =======================
# Evaluation
# =======================
def evaluate(student_model, teacher_model, val_loader, device, writer=None, epoch=None):
    student_model.eval()
    all_pixel_scores, all_pixel_labels = [], []
    all_img_scores, all_img_labels = [], []

    with torch.no_grad():
        for imgs, (img_labels, pixel_masks) in val_loader:
            imgs, pixel_masks = imgs.to(device), pixel_masks.to(device)
            anomaly = anomaly_map(imgs, teacher_model, student_model)

            all_pixel_scores.append(anomaly.cpu().numpy().ravel())
            all_pixel_labels.append(pixel_masks.cpu().numpy().ravel())

            img_scores = anomaly.view(anomaly.size(0), -1).max(dim=1)[0]
            all_img_scores.append(img_scores.cpu().numpy())
            all_img_labels.append(img_labels.numpy())

        # 可視化前 4 張
        if writer is not None and epoch is not None:
            val_imgs_vis, (val_labels_vis, val_masks_vis) = next(iter(val_loader))
            val_imgs_vis = val_imgs_vis.to(device)
            anomaly_vis = anomaly_map(val_imgs_vis, teacher_model, student_model)
            anomaly_norm = (anomaly_vis - anomaly_vis.min()) / (anomaly_vis.max() - anomaly_vis.min() + 1e-8)

            input_grid = make_grid(val_imgs_vis[:4], nrow=4)
            anomaly_grid = make_grid(anomaly_norm[:4].repeat(1,3,1,1), nrow=4)
            mask_grid = make_grid(val_masks_vis[:4].repeat(1,3,1,1), nrow=4)

            writer.add_image("Val/Input", input_grid, epoch)
            writer.add_image("Val/AnomalyMap", anomaly_grid, epoch)
            writer.add_image("Val/GT_Mask", mask_grid, epoch)

    all_pixel_scores = np.concatenate(all_pixel_scores)
    all_pixel_labels = np.concatenate(all_pixel_labels)
    all_img_scores = np.concatenate(all_img_scores)
    all_img_labels = np.concatenate(all_img_labels)

    pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
    img_auroc = roc_auc_score(all_img_labels, all_img_scores)
    return pixel_auroc, img_auroc

# =======================
# Main Pipeline
# =======================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load datasets
    train_dataset = MVTecDataset(root="./mvtec_ad", category="bottle", split="train", resize=256)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataset = MVTecDataset(root="./mvtec_ad", category="bottle", split="test", resize=256)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Load teacher
    teacher_ckpt = torch.load("draem_trained.pckl", map_location=device)
    teacher_model = ReconstructiveSubNetwork(in_channels=3, out_channels=3).to(device)
    teacher_model.load_state_dict(teacher_ckpt['reconstructive'])
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Student model
    student_model = ReconstructiveSubNetwork(in_channels=3, out_channels=3).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

    # TensorBoard
    writer = SummaryWriter(log_dir="./runs/rd4ad")
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_pixel_auroc = 0.0

    epochs = 50
    global_step = 0
    for epoch in range(epochs):
        student_model.train()
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                teacher_feats = get_embeddings(teacher_model, imgs)
            student_feats = get_embeddings(student_model, imgs)
            loss = rd4ad_loss(teacher_feats, student_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Train/Loss", loss.item(), global_step)
            global_step += 1

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {loss.item():.4f}")

        # Validation
        pixel_auroc, img_auroc = evaluate(student_model, teacher_model, val_loader, device, writer, epoch)
        writer.add_scalar("Val/Pixel_AUROC", pixel_auroc, epoch)
        writer.add_scalar("Val/Image_AUROC", img_auroc, epoch)
        print(f"Validation - Pixel AUROC: {pixel_auroc:.4f}, Image AUROC: {img_auroc:.4f}")

        # Save best checkpoint
        if pixel_auroc > best_pixel_auroc:
            best_pixel_auroc = pixel_auroc
            ckpt_path = os.path.join(checkpoint_dir, f"student_best_epoch{epoch+1}_pixelAUROC{pixel_auroc:.4f}.pth")
            torch.save({
                "epoch": epoch+1,
                "pixel_auroc": pixel_auroc,
                "img_auroc": img_auroc,
                "student_state_dict": student_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, ckpt_path)
            print(f"--> Saved best Student model to {ckpt_path}")

    writer.close()

    # --------------------------
    # Inference + visualization
    # --------------------------
    os.makedirs("./inference_results", exist_ok=True)
    student_model.eval()
    with torch.no_grad():
        for i, (imgs, (labels, masks)) in enumerate(val_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            anomaly = anomaly_map(imgs, teacher_model, student_model)
            anomaly_norm = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-8)
            anomaly_rgb = anomaly_norm.repeat(1,3,1,1)
            masks_rgb = masks.repeat(1,3,1,1)

            combined = torch.cat([imgs, anomaly_rgb, masks_rgb], dim=3)
            save_image(combined, f"./inference_results/comparison_batch{i+1}.png")
            print(f"Saved batch {i+1} comparison to ./inference_results/comparison_batch{i+1}.png")

    # --------------------------
    # Full evaluation report
    # --------------------------
    pixel_auroc, img_auroc = evaluate(student_model, teacher_model, val_loader, device)
    report_path = "./inference_results/rd4ad_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Pixel-level AUROC: {pixel_auroc:.4f}\n")
        f.write(f"Image-level AUROC: {img_auroc:.4f}\n")
    print(f"Evaluation done. Report saved to {report_path}")
    print(f"Pixel-level AUROC: {pixel_auroc:.4f}, Image-level AUROC: {img_auroc:.4f}")

# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    main()
