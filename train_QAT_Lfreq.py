# train_QAT_Lfeq.py (QAT + TensorBoardX & viz)

import torch
import torch.optim
import torch.quantization
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
import random
import numpy as np
import math
from tqdm import tqdm

import config as c
from datasets import Hinet_Dataset
from model_QAT import Model
from modules.Unet_common import DWT, IWT

# TensorBoardX 및 viz 로깅
from tensorboardX import SummaryWriter
import viz

# 재현성을 위한 시드 설정
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def computePSNR(origin, pred):
    """
    PSNR(Peak Signal-to-Noise Ratio) 계산 함수
    origin, pred: NumPy 배열, [H, W, C]
    """
    origin = origin.astype(np.float32)
    pred = pred.astype(np.float32)
    mse = np.mean((origin - pred) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10((255.0 ** 2) / mse)


def main_qat():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DWT/IWT 모듈 초기화 (GPU)
    dwt = DWT().to(device)
    iwt = IWT().to(device)

    # 데이터 로더
    transform = T.Compose([
        T.RandomCrop(c.cropsize),
        T.ToTensor()
    ])
    train_dataset = Hinet_Dataset(transforms_=transform, mode="train")
    trainloader = DataLoader(
        train_dataset,
        batch_size=c.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True
    )

    # 모델 불러오기 및 QAT 설정
    net = Model(c.init_model_path)
    net.to(device)  # 모델을 먼저 GPU로 올립니다
    net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    net.train()  # QAT 준비 전 반드시 train() 모드로 설정
    torch.quantization.prepare_qat(net, inplace=True)
    print("Model prepared for QAT.")

    optim = torch.optim.Adam(net.parameters(), lr=c.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=c.milestones, gamma=0.5
    )
    loss_fn = nn.L1Loss().to(device)

    # TensorBoardX writer 초기화
    writer = SummaryWriter(comment='hinet_qat', filename_suffix='QAT_Lfreq')

    # QAT 미세조정
    print("Starting QAT fine-tuning...")
    qat_epochs = getattr(c, 'qat_epochs', 10)
    for epoch in range(qat_epochs):
        net.train()
        for batch_idx, data in enumerate(
            tqdm(trainloader, desc=f"QAT Epoch {epoch+1}/{qat_epochs}")
        ):
            cover = data['cover_image'].to(device)
            secret = data['secret_image'].to(device)
            optim.zero_grad()

            # Forward (DWT 적용)
            cover_dwt = dwt(cover)
            secret_dwt = dwt(secret)
            input_dwt = torch.cat((cover_dwt, secret_dwt), dim=1)
            output_dwt = net(input_dwt)

            # Reverse pass
            # DWT 결과는 3채널 * 4 sub-bands = 12채널
            output_steg_dwt = output_dwt[:, :12, :, :]
            rev_output = net(output_dwt, rev=True)
            output_secret_dwt = rev_output[:, 12:, :, :]

            steg_img = iwt(output_steg_dwt)
            secret_rev = iwt(output_secret_dwt)

            # 손실 계산
            guide_loss = loss_fn(steg_img, cover)
            recon_loss = loss_fn(secret_rev, secret)
            cover_low = cover_dwt[:, :3, :, :]
            steg_low = output_steg_dwt[:, :3, :, :]
            low_freq_loss = loss_fn(steg_low, cover_low)
            total_loss = (
                c.lamda_reconstruction * recon_loss +
                c.lamda_guide * guide_loss +
                c.lamda_low_frequency * low_freq_loss
            )

            # Backprop
            total_loss.backward()
            optim.step()

            # 전역 스텝 계산
            global_step = epoch * len(trainloader) + batch_idx
            writer.add_scalar('Train/Loss', total_loss.item(), global_step)

            # 배치의 첫 번째 샘플로 PSNR 계산
            with torch.no_grad():
                steg_np = steg_img[0].detach().cpu().numpy().transpose(1,2,0) * 255.0
                cover_np = cover[0].detach().cpu().numpy().transpose(1,2,0) * 255.0
                secret_rev_np = secret_rev[0].detach().cpu().numpy().transpose(1,2,0) * 255.0
                secret_np = secret[0].detach().cpu().numpy().transpose(1,2,0) * 255.0
                psnr_c = computePSNR(cover_np, steg_np)
                psnr_s = computePSNR(secret_np, secret_rev_np)
            writer.add_scalar('Train/PSNR_C', psnr_c, global_step)
            writer.add_scalar('Train/PSNR_S', psnr_s, global_step)

            # viz 실시간 시각화
            viz.show_loss([total_loss.item(), math.log10(optim.param_groups[0]['lr'])])

        scheduler.step()
        print(f"Epoch {epoch+1}/{qat_epochs} - Loss: {total_loss.item():.4f}")

    # 모델 양자화 변환 및 저장
    print("Converting to INT8 model...")
    net.eval()
    net.to('cpu')
    quantized_model = torch.quantization.convert(net, inplace=True)
    save_path = os.path.join(c.MODEL_PATH, 'hinet_qat_quantized.pt')
    torch.save(quantized_model.state_dict(), save_path)
    print(f"Quantized model saved to {save_path}")

    writer.close()
    viz.signal_stop()


if __name__ == "__main__":
    main_qat()
