import torch
import torch.nn as nn
import torch.nn.quantized as nnq
from rrdb_denselayer import ResidualDenseBlock_out

class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, channel_in=3, channel_out=3, block_num=1):
        """
        Invertible Block(가역 블록)을 초기화합니다.
        """
        super(INV_block, self).__init__()

        self.split_len1 = channel_in // 2
        self.split_len2 = channel_in - self.split_len1

        self.f = subnet_constructor(self.split_len2, self.split_len1)
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        self.y = subnet_constructor(self.split_len1, self.split_len2)

        # ===== 코드 수정 시작: 연산자를 'functional' 하나로 통일 =====
        # 가중치 파일에 저장된 'functional' 키와 일치시키기 위해 단일 인스턴스를 생성합니다.
        self.functional = nnq.FloatFunctional()
        # ===== 코드 수정 끝 =====

    def forward(self, x, rev=False):
        """
        가역 블록의 정방향 또는 역방향 연산을 수행합니다.
        """
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        # ===== 코드 수정 시작: 모든 연산을 self.functional로 수행 =====
        if not rev:
            y1 = self.functional.add(x1, self.f(x2))
            y2 = self.functional.add(x2, self.r(y1))
            y3 = self.functional.add(y1, self.y(y2))
            out = self.functional.cat([y3, y2], 1)
        else:
            # 역방향 연산 시에는 표준 뺄셈(-)을 사용합니다.
            y3, y2 = x1, x2
            y1 = y3 - self.y(y2)
            x2_rev = y2 - self.r(y1)
            x1_rev = y1 - self.f(x2_rev)
            out = torch.cat((x1_rev, x2_rev), 1)
        # ===== 코드 수정 끝 =====

        return out