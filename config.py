# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -5.5
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01
# config.py 파일에 추가할 내용

# QAT 미세 조정을 위한 학습률 감소 시점 (에포크)
# 예: 10 에포크 훈련 시, 5번째와 8번째 에포크가 끝나고 학습률 감소
milestones = [5, 8]
lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1
device_ids = [0]

# Train:
batch_size = 4
cropsize = 224
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 1024
batchsize_val = 2
shuffle_val = False
val_freq = 50


# Dataset
TRAIN_PATH = '/content/drive/MyDrive/Colab/HiNet_all/HiNet_all/DIV2K_train_HR/DIV2K_train_HR/'
VAL_PATH = '/content/drive/MyDrive/Colab/HiNet_all/HiNet_all/DIV2K_valid_HR/DIV2K_valid_HR/'
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = '/content/drive/MyDrive/Colab/HiNet_all/HiNet_all/HiNet/model/'
checkpoint_on_error = True
SAVE_freq = 50


IMAGE_PATH = '/content/drive/MyDrive/Colab/HiNet_all/HiNet_all/HiNet/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'

# Load:
suffix = 'model.pt'
init_model_path = MODEL_PATH + suffix
tain_next = False
trained_epoch = 0
