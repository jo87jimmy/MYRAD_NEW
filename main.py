import torch  # 引入 PyTorch
from torchvision.datasets import ImageFolder  # 用於影像資料夾的資料集
import numpy as np  # 數值計算套件
import random  # 亂數控制
import os  # 檔案系統操作
from torch.utils.data import DataLoader  # PyTorch 的資料載入器
import torch.backends.cudnn as cudnn  # CUDA cuDNN 加速
import argparse  # 命令列參數處理
from torch.nn import functional as F  # 引入 PyTorch 的函式介面

from torch import optim
# 新增熱力圖可視化所需的函式庫
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2  # 匯入 OpenCV，用於影像處理

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model_unet import ReconstructiveSubNetwork
from student_models import StudentReconstructiveSubNetwork
import torchvision.utils as vutils

def setup_seed(seed):
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保證結果可重現
    torch.backends.cudnn.benchmark = False  # 關閉自動最佳化搜尋

# === 3. 特徵提取 Hook ===
def get_embeddings(model, x):
    feats = []
    hooks = []

    def hook_fn(_, __, output):
        feats.append(output)

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn))

    _ = model(x)

    for h in hooks:
        h.remove()

    return feats
# RD4AD Loss (Cosine Similarity)
def rd4ad_loss(teacher_feats, student_feats,cos=None):
    loss = 0
    for t, s in zip(teacher_feats, student_feats):
        # normalize to unit length
        t = nn.functional.normalize(t.flatten(1), dim=1)
        s = nn.functional.normalize(s.flatten(1), dim=1)
        loss += torch.mean(1 - cos(t, s))
    return loss

def train(_arch_, _class_, epochs, save_pth_path):
    # 訓練流程主函數
    print(f"🔧 類別: {_class_} | Epochs: {epochs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 選擇運算裝置
    print(f"🖥️ 使用裝置: {device}")
    # === 1. 載入 Teacher (DRAEM 訓練好的模型) ===
    # 教師模型 (已載入權重並設為 eval 模式)
    teacher_model = ReconstructiveSubNetwork(in_channels=3,
                                             out_channels=3)  # 重建子網路

    # === Step 2: 載入 checkpoint ===
    teacher_model_ckpt = torch.load(
        "DRAEM_seg_large_ae_large_0.0001_800_bs8_bottle_.pckl",
        map_location=device,
        weights_only=True)  # 載入教師重建模型權重

    teacher_model.load_state_dict(teacher_model_ckpt)  # 將權重加載到模型

    # 重要：載入權重後再移到設備
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # 設為評估模式，不更新權重
    for p in teacher_model.parameters():
        p.requires_grad = False

    # 學生模型
    student_dropout_rate = 0.2  # Dropout 率，可調整
    student_model = StudentReconstructiveSubNetwork(
        in_channels=3, out_channels=3,
        dropout_rate=student_dropout_rate)  # 學生重建模型

    # === 4. RD4AD Loss (Cosine Similarity) ===
    cos = nn.CosineSimilarity(dim=1)


    # === 5. Optimizer ===
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

    # === 6. TensorBoard ===
    writer = SummaryWriter(log_dir="./runs/rd4ad")

    # === 7. 訓練 & 驗證流程 ===
    def anomaly_map(img):
        with torch.no_grad():
            t_feats = get_embeddings(teacher_model, img)
            s_feats = get_embeddings(student_model, img)

        score_maps = []
        for t, s in zip(t_feats, s_feats):
            t = nn.functional.normalize(t, dim=1)
            s = nn.functional.normalize(s, dim=1)
            diff = 1 - torch.mean(t * s, dim=1, keepdim=True)  # (B,1,H,W)
            diff = nn.functional.interpolate(diff, size=img.shape[2:], mode="bilinear")
            score_maps.append(diff)

        anomaly = torch.mean(torch.stack(score_maps), dim=0)
        return anomaly

    epochs = 50
    global_step = 0

    for epoch in range(epochs):
        student_model.train()
        for imgs, _ in train_loader:  # train_loader: 只含正常樣本
            imgs = imgs.to(device)

            with torch.no_grad():
                teacher_feats = get_embeddings(teacher_model, imgs)

            student_feats = get_embeddings(student_model, imgs)

            loss = rd4ad_loss(teacher_feats, student_feats, cos)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log to TensorBoard
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            global_step += 1

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

        # === 驗證 ===
        student_model.eval()
        with torch.no_grad():
            val_imgs, _ = next(iter(val_loader))  # val_loader: 含正常+異常
            val_imgs = val_imgs.to(device)

            anomaly = anomaly_map(val_imgs)

            # log 原圖 & anomaly map
            writer.add_images("Val/Input", val_imgs, epoch)
            writer.add_images("Val/AnomalyMap", (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-8), epoch)

    writer.close()


    print("訓練完成！")  # 訓練結束

if __name__ == '__main__':
    import argparse
    import pandas as pd
    import os
    import torch

    # 解析命令列參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='bottle', type=str)  # 訓練類別
    parser.add_argument('--epochs', default=25, type=int)  # 訓練回合數
    parser.add_argument('--arch', default='wres50', type=str)  # 模型架構
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)

    args = parser.parse_args()

    setup_seed(111)  # 固定隨機種子
    save_pth_path = f"pths/best_{args.arch}_{args.category}"

    # 建立輸出資料夾
    save_pth_dir = save_pth_path if save_pth_path else 'pths/best'
    os.makedirs(save_pth_dir, exist_ok=True)

    # 開始訓練，並接收最佳模型路徑與結果
    train(args.arch, args.category, args.epochs, save_pth_path)
