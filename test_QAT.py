import torch
import torch.nn as nn
import torch.quantization
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.transforms as T

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
        batch_size=c.batchsize_val, # config.py에서 설정된 배치 사이즈 사용
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
    
    # 결과 저장 폴더가 없으면 생성
    os.makedirs(c.IMAGE_PATH_cover, exist_ok=True)
    os.makedirs(c.IMAGE_PATH_secret, exist_ok=True)
    os.makedirs(c.IMAGE_PATH_steg, exist_ok=True)
    os.makedirs(c.IMAGE_PATH_secret_rev, exist_ok=True)
    
    global_img_idx = 0 # 전체 이미지 인덱스를 추적하기 위한 변수

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
            
            # --- 역방향 연산 ---
            # test_QAT.py에서 이미 역방향 연산 주석 처리되어 있었으나, 복원된 secret 이미지 저장을 위해 필요하면 주석 해제
            # output_z는 forward에서 모델이 생성한 잔차 부분입니다. 
            # test.py에서는 gauss_noise를 사용하지만, test_QAT.py에서 이 부분이 주석 처리되어 있으므로
            # 여기서는 역방향을 위해 output_z의 형태만 가져와 0으로 채워서 사용하거나, 
            # test.py처럼 gauss_noise를 추가할 수 있습니다. 
            # 그러나 QAT 환경에서는 양자화된 모델이 rev=True일 때 quant/dequant를 건너뛰므로 
            # output_z를 직접 사용해도 됩니다. 
            # 여기서는 원본 test_QAT.py의 주석처리된 부분을 참조하여 수정합니다.
            
            # test_QAT.py 원본에서 주석 처리된 부분:
            # output_z = output_dwt.narrow(1, 4 * c.channels_in, output_dwt.shape[1] - 4 * c.channels_in)
            # output_z = gauss_noise(output_z.shape) # test.py에서는 사용

            # 역방향 연산을 위해 output_steg와 output_z를 결합
            # test_QAT.py 원본 코드의 주석 처리된 부분에 따라 output_z는 사용하지 않고 있었지만,
            # 완전한 역방향 복원을 위해서는 output_z를 forward에서 나온 결과 그대로 또는 노이즈를 넣어 사용해야 합니다.
            # 여기서는 test.py와 유사하게 forward에서 나온 output_z를 그대로 사용한다고 가정합니다.
            
            # test.py의 gauss_noise 함수가 device를 따르도록 수정되었으므로, 여기에도 적용
            output_z_for_rev = output_dwt.narrow(1, 4 * c.channels_in, output_dwt.shape[1] - 4 * c.channels_in)
            # test.py에 있는 gauss_noise 함수는 여기에 정의되지 않았으므로, 
            # test_QAT.py의 기존 로직을 따라 output_z_for_rev를 직접 사용하거나, 
            # test.py의 gauss_noise 정의를 가져와야 합니다. 
            # 일단은 test.py의 동작을 모방하기 위해 gauss_noise를 적용합니다.
            # gauss_noise 함수는 test_QAT.py에 정의되어 있지 않으므로 Unet_common에서 가져온 것처럼 직접 정의하거나, 
            # test.py에서 가져와야 합니다. 여기서는 test.py에서 가져온다고 가정합니다.
            
            # test.py에서 gauss_noise 함수 복사 (device 적용)
            def gauss_noise(shape):
                noise = torch.zeros(shape).to(device)
                for _ in range(noise.shape[0]):
                    noise = torch.randn(shape).to(device) # 배치 전체에 노이즈를 생성
                return noise
            
            # output_z_guass = gauss_noise(output_z_for_rev.shape) # 필요하다면 노이즈 추가
            # output_rev = torch.cat((output_steg_dwt, output_z_guass), 1)

            # 원본 test_QAT.py의 주석 처리된 역방향 연산 부분을 복구하고, test.py의 논리를 따릅니다.
            # test_QAT.py에서는 rev=True일 때 net(output_dwt, rev=True)만 호출합니다.
            # 이는 output_dwt 전체를 역방향 모델에 넣어 secret_rev를 얻는다는 것을 의미합니다.
            rev_output = net_int8(output_dwt, rev=True) # output_dwt 자체가 전체 인코딩 결과이므로 이를 역변환
            output_secret_dwt = rev_output[:, 12:, :, :] # secret_dwt 부분 추출
            secret_rev = iwt(output_secret_dwt) # 역 DWT 적용
            
            # 배치 내 각 이미지를 개별 파일로 저장
            for batch_idx in range(cover.shape[0]):
                current_idx = global_img_idx + batch_idx

                # 원본 이미지 저장 (선택 사항, 필요 시 주석 해제)
                save_image(cover[batch_idx], f"{c.IMAGE_PATH_cover}/cover_{current_idx:05d}.png")
                save_image(secret[batch_idx], f"{c.IMAGE_PATH_secret}/secret_{current_idx:05d}.png")
                
                # 스테고 이미지 저장
                save_image(steg_img[batch_idx], f"{c.IMAGE_PATH_steg}/steg_{current_idx:05d}.png")
                
                # 복원된 비밀 이미지 저장
                save_image(secret_rev[batch_idx], f"{c.IMAGE_PATH_secret_rev}/secret_rev_{current_idx:05d}.png")
        
        global_img_idx += cover.shape[0] # 다음 배치를 위해 전역 인덱스 업데이트

    print("Testing finished. Images saved to:", c.IMAGE_PATH)


if __name__ == "__main__":
    # config.py의 init_model_path 설정을 확인하여 양자화된 모델 경로를 정확히 지정하세요.
    # 예: c.init_model_path = os.path.join(c.MODEL_PATH, 'hinet_qat_quantized.pt')
    main_test_quantized()
