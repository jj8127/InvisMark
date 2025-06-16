import torch
import torch.nn as nn
from hinet import Hinet
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self, init_model_path=None):
        super(Model, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.hinet = Hinet()
        self.dequant = torch.quantization.DeQuantStub()

        if init_model_path:
            checkpoint = torch.load(init_model_path)
            
            state_dict = checkpoint['net']
            
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
            
            self.hinet.load_state_dict(new_state_dict)

    def forward(self, x, rev=False):
        # ===== 코드 수정 시작 =====
        # 역방향(rev=True) 연산 시에는 양자화를 적용하지 않도록 변경합니다.
        # 이렇게 하면 'sub' (뺄셈) 연산에서 발생하는 오류를 우회할 수 있습니다.
        if rev:
            # 양자화/역양자화 과정 없이 hinet을 직접 호출합니다.
            x = self.hinet(x, rev)
            return x
        # ===== 코드 수정 끝 =====

        # 정방향 연산 시에만 양자화를 적용합니다.
        x = self.quant(x)
        x = self.hinet(x, rev)
        x = self.dequant(x)
        return x