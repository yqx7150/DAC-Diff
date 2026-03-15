from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np

# import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
from scipy.io import loadmat

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation as controllable_generation
from utils import restore_checkpoint

sns.set(font_scale=2)
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
import cv2

from matplotlib.image import imread

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (
    ReverseDiffusionPredictor,
    LangevinCorrector,
    EulerMaruyamaPredictor,
    AncestralSamplingPredictor,
    NoneCorrector,
    NonePredictor,
    AnnealedLangevinDynamics,
)
import datasets
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse

# from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


import math
import scipy.io as io

from configs.ve import SIAT_kdata_ncsnpp as configs  # 修改config

# @title Load the score-based model
sde = "VESDE"  # @param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}


def save_img(img, img_path):
    img = np.clip(img * 255, 0, 255)  ##最小值0, 最大值255
    cv2.imwrite(img_path, img)


def write_Data(filedir, psnr, ssim):
    with open(filedir, "a+") as f:  # a+
        f.writelines(str(round(psnr, 4)) + "  " + str(round(ssim, 4)))
        f.write("\n")


def get_unet_input(list, length=512, low_detector=64):
    low_data = np.zeros((low_detector, length))
    for i in range(low_detector):
        low_data[i, :] = list[i, :]
    resized_low = np.array(
        cv2.resize(low_data, (length, length), interpolation=cv2.INTER_NEAREST)
    )

    return torch.tensor(resized_low, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


if sde.lower() == "vesde":
    for temp in range(38, 39):  # (24, 37) (37,51)   # (42, 46) (46,50)
     
        ckpt_filename = "/home/b110/code/LLX/lijiao_1_17/code_1_17/code_1_17/exp_1_17_f1_1500/checkpoints/checkpoint_4.pth"
        print(ckpt_filename)

        if not os.path.exists(ckpt_filename):
            print("!!!!!!!!!!!!!!" + ckpt_filename + " not exists")
            assert False
        config = configs.get_config()
        sde = VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
        batch_size = 1  # @param {"type":"integer"}
        config.training.batch_size = batch_size
        config.eval.batch_size = batch_size

        random_seed = 0  # @param {"type": "integer"}

        sigmas = mutils.get_sigmas(config)
        scaler = datasets.get_data_scaler(config)
        inverse_scaler = datasets.get_data_inverse_scaler(config)
        score_model = mutils.create_model(config)

        optimizer = get_optimizer(config, score_model.parameters())
        ema = ExponentialMovingAverage(
            score_model.parameters(), decay=config.model.ema_rate
        )
        state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)

        state = restore_checkpoint(ckpt_filename, state, config.device)
        ema.copy_to(score_model.parameters())

        # @title PC inpainting

        predictor = ReverseDiffusionPredictor  # @param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
        corrector = LangevinCorrector  # @param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
        snr = 0.21  # 0.07#0.075 #0.16 #@param {"type": "number"}
        n_steps = 1  # @param {"type": "integer"}
        probability_flow = False  # @param {"type": "boolean"}

        pc_inpainter = controllable_generation.get_pc_inpainter(
            sde,
            predictor,
            corrector,
            inverse_scaler,
            snr=snr,
            n_steps=n_steps,
            probability_flow=probability_flow,
            continuous=config.training.continuous,
            denoise=True,
        )

        path_70="/home/b110/code/LLX/lijiao_1_17/code_1_17/code_1_17/result_2_4/21845/"
        for num in [1,2,4,6,8]:
 
            file_path = os.path.join(path_70, str(num) + ".png")
            bad_input = cv2.imread(file_path,0)/255.
            dimg = sde.prior_sampling((1,256,256))  # 注意这里。尺寸
            dimg = dimg.squeeze(0).numpy()  # 原来有.permute(1,2,0)
            x_result = pc_inpainter(
                score_model, dimg  , bad_input,num
            )  # 预测器校正器操作
 
