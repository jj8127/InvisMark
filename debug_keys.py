# debug_keys.py

import torch
from collections import OrderedDict

# 프로젝트의 모델과 설정을 불러옵니다.
import config as c
from model_QAT import Model
from hinet import Hinet
from invblock import INV_block
from rrdb_denselayer import ResidualDenseBlock_out

def run_diagnostics():
    """
    모델이 기대하는 키와 파일에 실제 포함된 키를 비교하여 출력합니다.
    """
    print("=" * 60)
    print("Key-Mismatch-Finder: Model and Weight File Diagnostic")
    print("=" * 60)

    # --- 1. 모델이 기대하는 키 목록 출력 ---
    try:
        print("\n[Phase 1] Analyzing the keys expected by the model...")
        # test_QAT.py와 동일한 방식으로 모델 구조를 생성합니다.
        net = Model()
        net.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        net.train()
        torch.quantization.prepare_qat(net, inplace=True)
        net_int8 = torch.quantization.convert(net)
        
        print("\n--- Model's Expected Keys (What `load_state_dict` is looking for) ---")
        model_keys = set(net_int8.state_dict().keys())
        for key in sorted(list(model_keys)):
            print(key)
        print("-" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Failed to create model structure: {e}")
        return

    # --- 2. 파일에 실제 들어있는 키 목록 출력 ---
    try:
        print(f"\n[Phase 2] Analyzing the keys inside the file: '{c.init_model_path}'...")
        state_dict = torch.load(c.init_model_path, map_location='cpu')

        if not isinstance(state_dict, dict):
            print("\n[ERROR] The loaded file is not a valid state dictionary.")
            return

        print("\n--- File's Actual Keys (What is actually in the .pt file) ---")
        file_keys = set(state_dict.keys())
        if not file_keys:
            print("The state dictionary in the file is EMPTY!")
        for key in sorted(list(file_keys)):
            print(key)
        print("-" * 60)

    except FileNotFoundError:
        print(f"\n[ERROR] Weight file not found at path: {c.init_model_path}")
        return
    except Exception as e:
        print(f"\n[ERROR] Failed to load the weight file: {e}")
        return

    # --- 3. 차이점 비교 및 결론 ---
    print("\n[Phase 3] Diagnosis Result...")
    
    # 모델에는 있지만 파일에는 없는 키
    missing_in_file = model_keys - file_keys
    if missing_in_file:
        print("\n>>> CRITICAL: The following keys are EXPECTED by the model but are MISSING from the file:")
        for key in sorted(list(missing_in_file)):
            print(f"  - {key}")

    # 파일에는 있지만 모델에는 없는 키
    extra_in_file = file_keys - model_keys
    if extra_in_file:
        print("\n>>> INFO: The following keys are in the file but NOT a part of the current model structure:")
        for key in sorted(list(extra_in_file)):
            print(f"  - {key}")
            
    if not missing_in_file and not extra_in_file:
         print("\n>>> SUCCESS: All keys seem to match! If you still get an error, it might be a shape mismatch.")
    
    print("\n" + "="*60)
    print("Diagnosis complete. Compare the lists above to find the exact naming mismatch.")
    print("="*60)


if __name__ == '__main__':
    run_diagnostics()