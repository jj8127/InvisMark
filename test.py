import math
import torch
import torch.nn as nn
import torch.quantization
import torchvision
import numpy as np
# QAT 모델을 불러옵니다.
from model_QAT import Model
import config as c
import datasets
import modules.Unet_common as common
import os
from collections import OrderedDict

# 양자화된 모델은 CPU에서 실행하는 것이 일반적입니다.
device = torch.device("cpu")

def gauss_noise(shape):
    # .cuda() 호출을 제거하고 device를 사용합니다.
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)
    return noise

def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


# --- 새로운 모델 로딩 로직 ---
print("Loading Quantized Model for Testing...")
# 1. QAT 모델의 뼈대를 만듭니다. (기본적으로 training 모드)
net = Model(init_model_path=None)

# 2. 양자화 준비를 위해 training 모드에서 prepare_qat를 호출합니다.
net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(net, inplace=True)

# 3. 모델을 최종 양자화된 구조로 변환합니다.
torch.quantization.convert(net, inplace=True)

# 4. 이제 추론(테스트)을 할 것이므로, 모델을 evaluation 모드로 설정합니다.
net.eval()

# 5. 양자화된 모델의 가중치 파일을 불러옵니다.
# config.py의 suffix를 'hinet_qat_quantized.pt'로 변경하거나, 아래 파일명을 직접 사용하세요.
quantized_model_path = os.path.join(c.MODEL_PATH, 'hinet_qat_quantized.pt') 
print(f"Loading weights from: {quantized_model_path}")
state_dict = torch.load(quantized_model_path, map_location=device)
net.load_state_dict(state_dict)

net.to(device)

# --- 테스트 실행 로직 ---
dwt = common.DWT()
iwt = common.IWT()

# 결과 저장 폴더가 없으면 생성
os.makedirs(c.IMAGE_PATH_cover, exist_ok=True)
os.makedirs(c.IMAGE_PATH_secret, exist_ok=True)
os.makedirs(c.IMAGE_PATH_steg, exist_ok=True)
os.makedirs(c.IMAGE_PATH_secret_rev, exist_ok=True)

print("Starting inference...")
with torch.no_grad():
    # ===== 코드 수정 부분 시작 =====
    for i, data_dict in enumerate(datasets.testloader, 0):
        # 데이터로더가 반환하는 딕셔너리에서 cover와 secret 이미지를 추출합니다.
        cover = data_dict['cover_image'].to(device)
        secret = data_dict['secret_image'].to(device)
    # ===== 코드 수정 부분 끝 =====
            
        cover_input = dwt(cover)
        secret_input = dwt(secret)
        input_img = torch.cat((cover_input, secret_input), 1)

        #################
        #    forward:   #
        #################
        output = net(input_img)
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
        steg_img = iwt(output_steg)
        backward_z = gauss_noise(output_z.shape)

        #################
        #   backward:   #
        #################
        output_rev = torch.cat((output_steg, backward_z), 1)
        bacward_img = net(output_rev, rev=True)
        secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)
        cover_rev = bacward_img.narrow(1, 0, 4 * c.channels_in)
        cover_rev = iwt(cover_rev)
        
        # 중간 결과 저장
        # torchvision.utils.save_image(cover, os.path.join(c.IMAGE_PATH_cover, '%.5d.png' % i))
        # torchvision.utils.save_image(secret, os.path.join(c.IMAGE_PATH_secret, '%.5d.png' % i))
        torchvision.utils.save_image(steg_img, os.path.join(c.IMAGE_PATH_steg, '%.5d.png' % i))
        torchvision.utils.save_image(secret_rev, os.path.join(c.IMAGE_PATH_secret_rev, '%.5d.png' % i))
        
        print(f"Processed and saved image {i}")

print("Testing finished. Images saved to:", c.IMAGE_PATH)
