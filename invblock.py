# test_QAT.py (Final Corrected Version)

import torch
import torch.nn as nn
# 양자화를 위해 임포트
import torch.quantization
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.transforms as T

import os
from tqdm import tqdm
import numpy as np
import math

import config as c
from datasets import Hinet_Dataset
from modules.Unet_common import DWT, IWT


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


def main_test_quantized():
    # 양자화 모델 테스트는 주로 CPU에서 진행합니다.
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 데이터 로더
    transform = T.Compose([
        T.CenterCrop(c.cropsize_val),
        T.ToTensor(),
    ])
    test_dataset = Hinet_Dataset(transforms_=transform, mode="val")
    testloader = DataLoader(
        test_dataset,
        batch_size=c.batchsize_val,
        shuffle=False,
        pin_memory=True,
        num_workers=0, # Colab에서는 0으로 설정
        drop_last=False
    )

    # =================================================================
    # [핵심 수정] 양자화된 모델을 올바르게 불러오는 로직
    # =================================================================
    print("Loading Quantized Model for Testing...")
    
    # 1. 모델 객체를 생성 (QuantStub/DeQuantStub이 포함된 버전)
    #    이 모델은 아직 FP32 상태입니다.
    net = Model()
    net.eval()

    # 2. qconfig 설정 및 양자화 모델 '틀' 준비
    #    CPU 추론을 위한 'qnnpack' 백엔드 사용
    net.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # 모델을 양자화할 준비를 합니다 (옵저버 삽입). 
    # test 단계에서는 실제로 훈련하지 않으므로, eval 모드에서 prepare를 수행합니다.
    torch.quantization.prepare(net, inplace=True)

    # 3. convert를 통해 최종 INT8 모델 구조로 변환
    #    이 단계까지는 파라미터가 비어있는 INT8 모델의 '틀'만 생성된 상태
    net_int8 = torch.quantization.convert(net)
    print("INT8 model structure created.")

    # 4. 저장된 양자화 모델의 state_dict를 불러옴
    #    config.py의 init_model_path가 QAT로 훈련된 모델을 가리켜야 함
    print(f"Loading weights from: {c.init_model_path}")
    # CPU에서 모델을 불러옵니다.
    state_dict = torch.load(c.init_model_path, map_location='cpu')
    net_int8.load_state_dict(state_dict)
    
    # 최종적으로 모델을 평가 모드로 설정
    net_int8.eval()
    net_int8.to(device)

    # =================================================================
    # 테스트 루프
    # =================================================================
    dwt = DWT().to(device)
    iwt = IWT().to(device)
    
    for i, data in enumerate(tqdm(testloader, desc="Testing")):
        cover = data['cover_image'].to(device)
        secret = data['secret_image'].to(device)
        
        with torch.no_grad():
            # Forward pass
            cover_dwt = dwt(cover)
            secret_dwt = dwt(secret)
            input_dwt = torch.cat((cover_dwt, secret_dwt), 1)
            
            # net_int8 모델을 사용
            output_dwt = net_int8(input_dwt)
            
            # Reverse pass
            output_steg_dwt = output_dwt[:, :12, :, :]
            rev_output = net_int8(output_dwt, rev=True)
            output_secret_dwt = rev_output[:, 12:, :, :]
            
            # 결과 이미지 변환
            steg_img = iwt(output_steg_dwt)
            secret_rev = iwt(output_secret_dwt)
            
            # 결과 저장
            if not os.path.exists(c.IMAGE_PATH_steg):
                os.makedirs(c.IMAGE_PATH_steg)
            if not os.path.exists(c.IMAGE_PATH_secret_rev):
                os.makedirs(c.IMAGE_PATH_secret_rev)

            save_image(steg_img, f"{c.IMAGE_PATH_steg}/steg_{i}.png")
            save_image(secret_rev, f"{c.IMAGE_PATH_secret_rev}/secret_rev_{i}.png")

    print("Testing finished.")


if __name__ == "__main__":
    main_test_quantized()
