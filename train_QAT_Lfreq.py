import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.quantization
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import config as c
from datasets import Hinet_Dataset
from model_QAT import Model
from modules.Unet_common import DWT, IWT
import viz

# 재현성을 위한 시드 설정
def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# PSNR 계산 함수
def computePSNR(origin, pred):
    origin = origin.astype(np.float32)
    pred = pred.astype(np.float32)
    mse = np.mean((origin - pred) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10((255.0 ** 2) / mse)

# 메트릭 시각화 함수
def plot_metrics(steps, losses, psnr_c, psnr_s, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='.', linestyle='-')
    plt.title('Training Loss over Steps')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()

    # PSNR_C
    plt.figure(figsize=(10, 5))
    plt.plot(steps, psnr_c, marker='.', linestyle='-')
    plt.title('PSNR_C over Steps')
    plt.xlabel('Step')
    plt.ylabel('PSNR (Cover)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_psnr_c.png'))
    plt.close()

    # PSNR_S
    plt.figure(figsize=(10, 5))
    plt.plot(steps, psnr_s, marker='.', linestyle='-')
    plt.title('PSNR_S over Steps')
    plt.xlabel('Step')
    plt.ylabel('PSNR (Secret)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_psnr_s.png'))
    plt.close()


def main_qat():
    set_seed(c.seed if hasattr(c, 'seed') else 100)
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
    net.to(device)
    net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    net.train()
    torch.quantization.prepare_qat(net, inplace=True)
    print("Model prepared for QAT.")

    optim = torch.optim.Adam(net.parameters(), lr=c.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=c.milestones, gamma=0.5
    )
    loss_fn = nn.L1Loss().to(device)

    # TensorBoardX writer 초기화
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    log_dir = os.path.join(script_dir, 'runs')
    writer = SummaryWriter(logdir=log_dir, comment='hinet_qat_Lfreq')

    # 메트릭 저장용 리스트
    steps, losses, psnr_c_list, psnr_s_list = [], [], [], []

    qat_epochs = getattr(c, 'qat_epochs', 10)
    global_step = 0

    print("Starting QAT fine-tuning...")
    for epoch in range(qat_epochs):
        net.train()
        for batch_idx, data in enumerate(trainloader):
            cover = data['cover_image'].to(device)
            secret = data['secret_image'].to(device)
            optim.zero_grad()

            # Forward (DWT 적용)
            cover_dwt = dwt(cover)
            secret_dwt = dwt(secret)
            input_dwt = torch.cat((cover_dwt, secret_dwt), dim=1)
            output_dwt = net(input_dwt)

            # Reverse pass
            output_steg_dwt = output_dwt[:, :12, :, :]
            rev_output = net(output_dwt, rev=True)
            output_secret_dwt = rev_output[:, 12:, :, :]

            steg_img = iwt(output_steg_dwt)
            secret_rev = iwt(output_secret_dwt)

            # 손실 계산
            guide_loss = loss_fn(steg_img, cover)
            recon_loss = loss_fn(secret_rev, secret)
            low_freq_loss = loss_fn(output_steg_dwt[:, :3, :, :], cover_dwt[:, :3, :, :])
            total_loss = (
                c.lamda_reconstruction * recon_loss +
                c.lamda_guide * guide_loss +
                c.lamda_low_frequency * low_freq_loss
            )

            total_loss.backward()
            optim.step()

            # 로그 업데이트
            psnr_c = computePSNR((cover.detach().cpu().numpy().transpose(0,2,3,1)[0] * 255),
                                  (steg_img.detach().cpu().numpy().transpose(0,2,3,1)[0] * 255))
            psnr_s = computePSNR((secret.detach().cpu().numpy().transpose(0,2,3,1)[0] * 255),
                                  (secret_rev.detach().cpu().numpy().transpose(0,2,3,1)[0] * 255))

            global_step += 1
            steps.append(global_step)
            losses.append(total_loss.item())
            psnr_c_list.append(psnr_c)
            psnr_s_list.append(psnr_s)

            writer.add_scalar('Train/Loss', total_loss.item(), global_step)
            writer.add_scalar('Train/PSNR_C', psnr_c, global_step)
            writer.add_scalar('Train/PSNR_S', psnr_s, global_step)
            viz.show_loss([total_loss.item(), math.log10(optim.param_groups[0]['lr'])])

        scheduler.step()
        print(f"Epoch {epoch+1}/{qat_epochs} - Loss: {total_loss.item():.4f}")

    # 모델 양자화 및 저장
    print("Converting to INT8 model...")
    net.eval()
    net.to('cpu')
    quantized_model = torch.quantization.convert(net, inplace=True)
    save_path = os.path.join(c.MODEL_PATH, 'hinet_qat_quantized2.pt')
    torch.save(quantized_model.state_dict(), save_path)
    print(f"Quantized model saved to {save_path}")

    writer.close()
    viz.signal_stop()

    # 메트릭 플롯 저장
    plot_metrics(steps, losses, psnr_c_list, psnr_s_list, save_dir=script_dir)
    print(f"Metrics plots saved in {script_dir}")

if __name__ == "__main__":
    main_qat()
