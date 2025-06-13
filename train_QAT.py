# train_QAT.py

import torch
import torch.optim
# QAT를 위해 torch.quantization 임포트
import torch.quantization
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

import os
import random
import numpy as np
from tqdm import tqdm

import config as c
from datasets import Hinet_Dataset
from model import Model
from modules.Unet_common import DWT, IWT

# 시드 고정
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def main_qat():
    # GPU 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로더 준비 (원본 train.py와 동일)
    transform = T.Compose([
        T.RandomCrop(c.cropsize),
        T.ToTensor()
    ])
    train_dataset = Hinet_Dataset(transforms_=transform, mode="train")
    trainloader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    
    # 웨이블릿 변환 모듈
    dwt = DWT().to(device)
    iwt = IWT().to(device)

    # =================================================================
    # 1. 모델 준비 및 QAT 설정 (Model Preparation & QAT Setup)
    # =================================================================
    print("Loading pre-trained FP32 model for QAT...")
    # config.py의 init_model_path에 사전 훈련된 모델 경로를 지정해야 합니다.
    # (주의: 이 Model 클래스는 QuantStub/DeQuantStub이 추가된 버전이어야 합니다)
    net = Model(c.init_model_path)
    net.eval()  # QAT 준비 전에는 반드시 eval() 모드로 설정

    print("Preparing model for QAT...")
    # 양자화 설정(qconfig) 정의. 'fbgemm'은 x86 CPU를 위한 권장 백엔드
    net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # QAT를 위한 모델 준비: 모델 내 Conv2d 같은 레이어에 '옵저버'를 삽입함
    torch.quantization.prepare_qat(net, inplace=True)
    print("Model prepared for QAT.")

    net.to(device) # GPU로 보냄
    
    # 옵티마이저 및 스케줄러 설정
    optim = torch.optim.Adam(net.parameters(), lr=c.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=c.milestones, gamma=0.5)
    loss_reconstruction = nn.L1Loss().to(device)

    # =================================================================
    # 2. QAT 미세 조정 (QAT Fine-tuning)
    # =================================================================
    print("Starting QAT fine-tuning...")
    # QAT는 보통 더 적은 에포크로 미세조정합니다.
    qat_epochs = c.qat_epochs if hasattr(c, 'qat_epochs') else 10 # config에 qat_epochs를 추가하거나 기본값 10 사용

    for i_epoch in range(qat_epochs):
        net.train() # 미세조정을 위해 train() 모드로 설정
        
        # 훈련 루프는 원본과 거의 동일
        for i_batch, data in enumerate(tqdm(trainloader, desc=f"QAT Epoch {i_epoch + 1}/{qat_epochs}")):
            cover = data['cover_image'].to(device)
            secret = data['secret_image'].to(device)

            optim.zero_grad()
            
            # Forward pass
            cover_dwt = dwt(cover)
            secret_dwt = dwt(secret)
            
            input_dwt = torch.cat((cover_dwt, secret_dwt), 1)
            output_dwt = net(input_dwt)
            
            # Loss calculation
            output_steg_dwt = output_dwt[:, :12, :, :]
            output_secret_rev_dwt = net(output_steg_dwt, rev=True)
            
            steg_img = iwt(output_steg_dwt)
            secret_rev = iwt(output_secret_rev_dwt)
            
            guide_loss = loss_reconstruction(steg_img, cover)
            reconstruction_loss = loss_reconstruction(secret_rev, secret)
            
            total_loss = c.lamda_reconstruction * reconstruction_loss + c.lamda_guide * guide_loss
            
            # Backward and optimize
            total_loss.backward()
            optim.step()

        scheduler.step()
        print(f"QAT Epoch {i_epoch + 1} finished. Loss: {total_loss.item():.4f}")

    # =================================================================
    # 3. 양자화 모델 변환 및 저장 (Convert and Save Quantized Model)
    # =================================================================
    print("Fine-tuning finished. Converting to a quantized INT8 model...")
    net.eval()
    net.to('cpu') # 변환은 반드시 CPU에서 수행해야 함

    # QAT로 훈련된 모델을 실제 8-bit 정수 모델로 변환
    model_quantized = torch.quantization.convert(net, inplace=True)
    print("Model converted to INT8 successfully.")

    # 최종 양자화된 모델 저장
    save_path = os.path.join(c.MODEL_PATH, 'hinet_qat_quantized.pt')
    torch.save(model_quantized.state_dict(), save_path)
    print(f"Quantized model saved to {save_path}")

if __name__ == "__main__":
    main_qat()