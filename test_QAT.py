# test_quantized.py (Final Corrected Version)

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

import config as c
# 수정된 model_QAT.py 와 datasets.py 를 사용
from model_QAT import Model
from datasets import Hinet_Dataset
from modules.Unet_common import DWT, IWT


def main_test_quantized():
    # 추론 환경이므로 CPU 사용을 권장 (엣지 디바이스 환경 시뮬레이션)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 데이터 로더 (기존 test.py와 유사)
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
    # [핵심 수정] 양자화된 모델을 불러오는 로직
    # =================================================================
    print("Loading Quantized Model for Testing...")
    
    # 1. 모델 객체를 생성 (QuantStub/DeQuantStub이 포함된 버전)
    net = Model()

    # 2. qconfig 설정 및 prepare_qat (구조를 맞춰주기 위해 필요)
    #    CPU 추론을 위한 'qnnpack' 백엔드 사용
    net.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # [수정] prepare_qat를 호출하기 전에 반드시 모델을 훈련 모드로 설정합니다.
    net.train()
    torch.quantization.prepare_qat(net, inplace=True)

    # 3. convert를 통해 최종 INT8 모델 구조로 변환
    #    이 단계까지는 파라미터가 비어있는 INT8 모델의 '틀'만 생성된 상태
    net_int8 = torch.quantization.convert(net)
    print("INT8 model structure created.")

    # 4. 저장된 양자화 모델의 state_dict를 불러옴
    #    config.py의 init_model_path가 'hinet_qat_quantized.pt'를 가리켜야 함
    print(f"Loading weights from: {c.init_model_path}")
    # CPU에서 모델을 불러옵니다.
    state_dict = torch.load(c.init_model_path, map_location='cpu')
    net_int8.load_state_dict(state_dict)
    
    # 추론을 위해 최종적으로 eval() 모드로 설정
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
