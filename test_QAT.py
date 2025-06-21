# test_QAT.py (최초 사용자 제공 버전)

import torch
import torch.nn as nn
import torch.quantization
from torchvision.utils import save_image # 수정 전에는 torchvision.utils.save_image 임포트 되어 있었습니다.
from torch.utils.data import DataLoader # 수정 전에는 torch.utils.data.DataLoader 임포트 되어 있었습니다.
import torchvision.transforms as T # 수정 전에는 torchvision.transforms 임포트 되어 있었습니다.

import os
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

import config as c
from model_QAT import Model
from datasets import Hinet_Dataset
from modules.Unet_common import DWT, IWT


def main_test_quantized():
    device = torch.device("cpu")
    print(f"Using device: {device}")

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
        num_workers=0,
        drop_last=False
    )

    print("Loading Quantized Model for Testing...")
    
    net = Model()
    net.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    net.train()
    torch.quantization.prepare_qat(net, inplace=True)

    net_int8 = torch.quantization.convert(net)
    print("INT8 model structure created.")

    print(f"Loading weights from: {c.init_model_path}")
    state_dict = torch.load(c.init_model_path, map_location='cpu')
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.model.'):
            name = k[13:]
        elif k.startswith('model.'):
            name = k[6:]
        elif k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    
    net_int8.load_state_dict(new_state_dict, strict=False)
    
    net_int8.eval()
    net_int8.to(device)

    dwt = DWT().to(device)
    iwt = IWT().to(device)
    
    # 결과 저장 폴더가 없으면 생성 (이 부분은 이전 대화에서 제가 추가했거나 test.py에 있던 내용이 반영된 것일 수 있습니다)
    # 원본 test_QAT.py에는 이 부분이 없었을 가능성도 있습니다.
    if not os.path.exists(c.IMAGE_PATH_steg):
        os.makedirs(c.IMAGE_PATH_steg)
    # if not os.path.exists(c.IMAGE_PATH_secret_rev): # 이 라인도 주석 처리되어 있었습니다.
    #     os.makedirs(c.IMAGE_PATH_secret_rev)


    for i, data in enumerate(tqdm(testloader, desc="Testing")):
        cover = data['cover_image'].to(device)
        secret = data['secret_image'].to(device)
        
        with torch.no_grad():
            # --- 정방향 연산 ---
            cover_dwt = dwt(cover)
            secret_dwt = dwt(secret)
            input_dwt = torch.cat((cover_dwt, secret_dwt), 1)
            
            output_dwt = net_int8(input_dwt)
            
            output_steg_dwt = output_dwt[:, :12, :, :]
            steg_img = iwt(output_steg_dwt)
            
            # ===== 코드 수정 시작: 오류가 발생하는 역방향 연산 주석 처리 =====
            # rev_output = net_int8(output_dwt, rev=True) # 이 라인이 주석 처리되어 있었습니다.
            # output_secret_dwt = rev_output[:, 12:, :, :] # 이 라인이 주석 처리되어 있었습니다.
            # secret_rev = iwt(output_secret_dwt) # 이 라인이 주석 처리되어 있었습니다.
            # ===== 코드 수정 끝 =====

            # 결과 저장
            save_image(steg_img, f"{c.IMAGE_PATH_steg}/steg_{i}.png")
            # save_image(secret_rev, f"{c.IMAGE_PATH_secret_rev}/secret_rev_{i}.png") # 이 라인이 주석 처리되어 있었습니다.

    print("Testing finished.")

# main_test_quantized 함수 호출 (이 부분은 수정 전에도 있었을 것입니다)
if __name__ == "__main__":
    main_test_quantized()
