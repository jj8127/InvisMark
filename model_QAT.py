import torch
import torch.nn as nn
from hinet import Hinet
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self, init_model_path=None):
        super(Model, self).__init__()
        # 1. QuantStub과 DeQuantStub을 추가합니다.
        self.quant = torch.quantization.QuantStub()
        self.hinet = Hinet()
        self.dequant = torch.quantization.DeQuantStub()

        if init_model_path:
            # 체크포인트(딕셔너리)를 불러옵니다.
            checkpoint = torch.load(init_model_path)
            
            # 딕셔너리에서 'net' 키를 사용해 실제 모델 가중치(state_dict)를 추출합니다.
            state_dict = checkpoint['net']
            
            # (DataParallel 접두사 제거) 새로운 state_dict를 만들어 키를 정리합니다.
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
              if k.startswith('module.model.'):
                  name = k[13:]  # 'module.model.' 접두사(13글자) 제거
              elif k.startswith('model.'):
                  name = k[6:]   # 'model.' 접두사(6글자) 제거
              elif k.startswith('module.'):
                  name = k[7:]   # 'module.' 접두사(7글자) 제거
              else:
                  name = k
              new_state_dict[name] = v
            
            # 정리된 가중치를 모델에 불러옵니다.
            self.hinet.load_state_dict(new_state_dict)

    def forward(self, x, rev=False):
        # 2. 양자화 시작과 끝을 지정합니다.
        x = self.quant(x)      # 입력값을 양자화 모드로 전환
        x = self.hinet(x, rev)  # HiNet 연산 수행
        x = self.dequant(x)    # 출력값을 다시 부동 소수점 모드로 전환
        return x