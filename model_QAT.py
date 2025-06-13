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
                name = k[13:] 
                new_state_dict[name] = v
            
            self.hinet.load_state_dict(new_state_dict)

    # 'def forward'는 __init__과 같은 레벨이므로, 앞에는 4칸(또는 1탭) 들여쓰기가 있어야 합니다.
    def forward(self, x, rev=False):
        x = self.quant(x)
        x = self.hinet(x, rev)
        x = self.dequant(x)
        return x