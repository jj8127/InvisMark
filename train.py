#!/usr/bin/env python
# train.py (origianl Version)

import torch
import torch.nn
import torch.optim
import math
import numpy as np
from model import *
import config as c
from tensorboardX import SummaryWriter
import datasets
import viz
import modules.Unet_common as common
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#역할: 지정된 shape에 맞춰 배치별 표준 정규분포 노이즈를 생성합니다.
#이유: 가우시안 노이즈를 잠재 공간(latent)에 주입해, 숨긴 비밀(secret) 정보 복원 시 불확실성을 모델링합니다.
def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

#손실함수 3개를 정의합니다.
def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


#네트워크 전체 파라미터 수와 학습 가능한 파라미터 수를 계산해, 모델 복잡도를 확인합니다.
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

#PSNR은 복원 이미지 품질 평가 척도로, 클수록 원본과 가까움을 의미합니다.
def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)

#체크포인트에서 모델과 옵티마이저 상태를 불러옵니다.
#'tmp_var' not in k 필터링은 임시 변수를 제외하기 위함입니다.
def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


#####################
# Model initialize: #
#####################
net = Model() #Model() 인스턴스 생성
net.cuda() #
init_model(net) #init_model로 가중치 초기화
net = torch.nn.DataParallel(net, device_ids=c.device_ids) #DataParallel으로 다중 GPU 지원
para = get_parameter_number(net)
print(para)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

#Adam 옵티마이저와 StepLR 스케줄러 사용.
#학습률, 베타, 감쇠율 등은 config.py에서 관리하여 재현성과 실험 편의성을 높입니다
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

dwt = common.DWT()
iwt = common.IWT()

#이어 학습(train_next) 여부에 따라 체크포인트 로드
#TensorBoardX 로깅을 위해 SummaryWriter 준비
if c.tain_next:
    load(c.MODEL_PATH + c.suffix)

try:
    writer = SummaryWriter(comment='hinet', filename_suffix="steg")

    for i_epoch in range(c.epochs):
        # Epoch 인덱스 조정
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []

        #################
        #     train:    #
        #################

        for i_batch, data in enumerate(datasets.trainloader):
            # 배치 분할: 절반은 커버 이미지, 절반은 비밀 이미지로 분리
            data = data.to(device)
            cover = data[data.shape[0] // 2:]
            secret = data[:data.shape[0] // 2]
            #Wavelet 변환
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            # 입력 이미지 결합: 커버와 비밀 이미지를 채널 차원에서 결합
            input_img = torch.cat((cover_input, secret_input), 1)

            #################
            #    forward:   #
            #################
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            steg_img = iwt(output_steg)

            #################
            #   backward:   #
            #################

            output_z_guass = gauss_noise(output_z.shape)

            output_rev = torch.cat((output_steg, output_z_guass), 1)
            output_image = net(output_rev, rev=True)

            secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev = iwt(secret_rev)

            #################
            #     loss:     #
            #################
            g_loss = guide_loss(steg_img.cuda(), cover.cuda())
            r_loss = reconstruction_loss(secret_rev, secret)
            steg_low = output_steg.narrow(1, 0, c.channels_in)
            cover_low = cover_input.narrow(1, 0, c.channels_in)
            l_loss = low_frequency_loss(steg_low, cover_low)

            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])

        #################
        #     val:    #
        #################
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                psnr_s = []
                psnr_c = []
                net.eval()
                for x in datasets.testloader:
                    x = x.to(device)
                    cover = x[x.shape[0] // 2:, :, :, :]
                    secret = x[:x.shape[0] // 2, :, :, :]
                    cover_input = dwt(cover)
                    secret_input = dwt(secret)

                    input_img = torch.cat((cover_input, secret_input), 1)

                    #################
                    #    forward:   #
                    #################
                    output = net(input_img)
                    output_steg = output.narrow(1, 0, 4 * c.channels_in)
                    steg = iwt(output_steg)
                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                    output_z = gauss_noise(output_z.shape)

                    #################
                    #   backward:   #
                    #################
                    output_steg = output_steg.cuda()
                    output_rev = torch.cat((output_steg, output_z), 1)
                    output_image = net(output_rev, rev=True)
                    secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                    secret_rev = iwt(secret_rev)

                    secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret = secret.cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    steg = steg.cpu().numpy().squeeze() * 255
                    np.clip(steg, 0, 255)
                    psnr_temp = computePSNR(secret_rev, secret)
                    psnr_s.append(psnr_temp)
                    psnr_temp_c = computePSNR(cover, steg)
                    psnr_c.append(psnr_temp_c)

                writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, i_epoch)
                writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)

        viz.show_loss(epoch_losses)
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)

        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i' % i_epoch + '.pt')
        weight_scheduler.step()

    torch.save({'opt': optim.state_dict(),
                'net': net.state_dict()}, c.MODEL_PATH + 'model' + '.pt')
    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, c.MODEL_PATH + 'model_ABORT' + '.pt')
    raise

finally:
    viz.signal_stop()
