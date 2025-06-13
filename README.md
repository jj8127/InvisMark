````markdown
# InvisMark 프로젝트 실행 가이드

이 문서는 InvisMark 비가시적 워터마킹 모델을 Quantization-Aware Training(QAT) 방식으로 학습 및 실행하는 전체 과정을 단계별로 안내합니다.

---

## 목차
1. [환경 및 전제 조건](#환경-및-전제-조건)
2. [저장소 클론 및 설치](#저장소-클론-및-설치)
3. [데이터셋 다운로드 및 구성](#데이터셋-다운로드-및-구성)
4. [디렉토리 구조 안내](#디렉토리-구조-안내)
5. [환경 변수 및 경로 설정](#환경-변수-및-경로-설정)
6. [QAT 학습 실행](#qat-학습-실행)
7. [모델 검증 및 추론](#모델-검증-및-추론)
8. [유용한 팁 및 주의사항](#유용한-팁-및-주의사항)

---

## 1. 환경 및 전제 조건
- **운영체제**: Linux / macOS (Ubuntu 20.04 권장)
- **Python 버전**: 3.8 이상 (3.8~3.10 테스트 완료)
- **CUDA & cuDNN**: GPU 사용 시 CUDA 11.1+, cuDNN 8 이상
- **디스크 용량**: 최소 20GB 여유 공간
- **Python 가상환경**: Conda 또는 venv 사용 권장

```bash
# 예시: venv 기반 가상환경 생성
python3 -m venv invismark_env
source invismark_env/bin/activate
````

---

## 2. 저장소 클론 및 설치

1. 원격 저장소 복제

   ```bash
   git clone https://github.com/your-username/InvisMark.git
   cd InvisMark
   ```

2. 주요 라이브러리 설치

   ```bash
   # PyTorch 및 Torchvision 설치 (CUDA 버전에 맞게 선택)
   pip install torch torchvision

   # 기타 의존 라이브러리
   pip install numpy pandas matplotlib scikit-image scikit-learn pywavelets \
               opencv-python tensorboardx torchsummary coloredlogs humanfriendly \
               ipdb natsort
   ```

---

## 3. 데이터셋 다운로드 및 구성

### DIV2K\_train\_HR (훈련 데이터)

```bash
# 약 7.1GB ZIP 파일 다운로드
wget -q https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
# 압축 해제
unzip -q DIV2K_train_HR.zip -d DIV2K_train_HR
```

### DIV2K\_valid\_HR (검증 데이터)

```bash
wget -q https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip -q DIV2K_valid_HR.zip -d DIV2K_valid_HR
```

---

## 4. 디렉토리 구조 안내

```
InvisMark/
├── data/
│   ├── DIV2K_train_HR/       # 훈련용 고해상도 이미지 디렉토리
│   └── DIV2K_valid_HR/       # 검증용 고해상도 이미지 디렉토리
├── model/                    # 학습된 모델 및 체크포인트
├── image/                    # 결과 이미지 저장
├── train_QAT_Lfreq.py        # QAT 학습 스크립트
├── train_QAT.py              # 일반 QAT 학습 스크립트
├── test_inference.py         # 추론 및 평가
└── README.md                 # 실행 가이드 (본 문서)
```

---

## 5. 환경 변수 및 경로 설정

스크립트 내부의 경로 변수를 사용자의 환경에 맞게 수정합니다.

```python
# train_QAT_Lfreq.py 등의 상단에 위치
TRAIN_PATH = '/절대/경로/To/InvisMark/DIV2K_train_HR/'
VAL_PATH   = '/절대/경로/To/InvisMark/DIV2K_valid_HR/'
MODEL_PATH = '/절대/경로/To/InvisMark/model/'
IMAGE_PATH = '/절대/경로/To/InvisMark/image/'

# Pretrained 모델 파일명 지정
init_model_path = MODEL_PATH + 'your_pretrained_model.pt'
```

> **Tip:** 절대 경로를 사용하면 경로 관련 오류를 줄일 수 있습니다.

---

## 6. QAT 학습 실행

1. 기본 QAT 학습

   ```bash
   python train_QAT.py \
       --train_dir $TRAIN_PATH \
       --val_dir   $VAL_PATH   \
       --save_dir  $MODEL_PATH \
       --init_model $init_model_path
   ```

2. Low-frequency 강조 QAT 학습

   ```bash
   python train_QAT_Lfreq.py \
       --train_dir $TRAIN_PATH \
       --val_dir   $VAL_PATH   \
       --save_dir  $MODEL_PATH \
       --init_model $init_model_path \
       --wavelet-level 2
   ```

> 각 옵션에 대한 설명은 스크립트 상단의 `--help`를 통해 확인할 수 있습니다.

```bash
python train_QAT_Lfreq.py --help
```

---

## 7. 모델 검증 및 추론

학습이 완료된 체크포인트를 사용하여 테스트 이미지를 처리합니다.

```bash
python test_inference.py \
    --checkpoint $MODEL_PATH/qat_final.pt \
    --input_dir  path/to/test_images \
    --output_dir $IMAGE_PATH/results
```

결과는 `IMAGE_PATH/results` 폴더에 저장되며, concealing/revealing된 이미지가 생성됩니다.

---

## 8. 유용한 팁 및 주의사항

* 학습 시 로그 확인: `tensorboard --logdir runs/` 명령으로 학습 과정을 모니터링하세요.
* GPU 메모리 부족 시: `--batch-size` 또는 `--wavelet-level`을 조정하여 메모리 사용량을 최적화하세요.
* 오류 발생 시: 경로 설정, 라이브러리 버전, GPU 드라이버(CUDA/cuDNN) 호환성을 먼저 점검하세요.
* 추가 문의 및 기여: [Issues](https://github.com/your-username/InvisMark/issues)에 이슈를 등록해주세요.

---

**감사합니다. 성공적인 학습과 실험 되시길 바랍니다!**

```
```
