# invblock.py (수정 완료된 코드)

from math import exp
import torch
import torch.nn as nn
# FloatFunctional을 임포트하여 양자화된 텐서 연산을 지원하도록 합니다.
from torch.ao.nn.quantized import FloatFunctional

import config as c
from rrdb_denselayer import ResidualDenseBlock_out


class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        
        # ρ, η, φ는 그대로 유지
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        self.f = subnet_constructor(self.split_len2, self.split_len1)

        # 양자화된 텐서 간의 연산을 위한 기능(functional) 래퍼(wrapper)를 인스턴스화합니다.
        # 덧셈, 곱셈, 연결(concatenation)에 사용합니다.
        self.functional = FloatFunctional()

    def e(self, s):
        # 이 부분은 dequantize -> 연산 -> quantize가 필요할 수 있으나, 일단 유지합니다.
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev: # 순방향 (Concealing)
            t2 = self.f(x2)
            # y1 = x1 + t2
            # FloatFunctional을 사용하여 양자화된 텐서 덧셈을 수행합니다.
            y1 = self.functional.add(x1, t2)
            
            s1, t1 = self.r(y1), self.y(y1)
            
            # y2 = self.e(s1) * x2 + t1
            # 곱셈과 덧셈 또한 FloatFunctional을 사용합니다.
            y2_scaled = self.functional.mul(self.e(s1), x2)
            y2 = self.functional.add(y2_scaled, t1)
            
            # 최종 결과를 cat으로 합칩니다.
            out = self.functional.cat([y1, y2], 1)

        else: # 역방향 (Revealing)
            s1, t1 = self.r(x1), self.y(x1)
            
            # [수정] FloatFunctional에 없는 div, sub 대신 일반 연산자 사용
            # y2 = (x2 - t1) / self.e(s1)
            y2_sub = x2 - t1
            y2 = y2_sub / self.e(s1)

            t2 = self.f(y2)
            
            # [수정] FloatFunctional에 없는 sub 대신 일반 연산자 사용
            # y1 = x1 - t2
            y1 = x1 - t2

            # 최종 결과를 cat으로 합칩니다.
            out = self.functional.cat([y1, y2], 1)
            
        return out
