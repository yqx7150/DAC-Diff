import os
from models import utils as mutils
import torch
import numpy as np
from sampling import (
    NoneCorrector,
    NonePredictor,
    shared_corrector_update_fn,
    shared_predictor_update_fn,
)
import functools
import cv2
import math
from scipy.io import savemat
# from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import scipy.io as io


# from lmafit_mc_adp_v2_numpy import lmafit_mc_adp


def write_Data(filedir, model_num, psnr, ssim):
    # filedir="result.txt"
    with open(os.path.join("./results", filedir), "a+") as f:  # a+
        f.writelines(
            str(model_num)
            + " "
            + "["
            + str(round(psnr, 2))
            + " "
            + str(round(ssim, 4))
            + "]"
        )
        f.write("\n")


def save_img(img, img_path):

    img = np.clip(img * 255, 0, 255)

    cv2.imwrite(img_path, img)


def write_Data(filedir, num, psnr, ssim):
    # filedir="result.txt"
    with open(os.path.join(filedir, str(num) + ".txt"), "a+") as f:  # a+
        f.writelines(str(round(psnr, 4)) + "  " + str(round(ssim, 4)))
        f.write("\n")


def get_pc_inpainter(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-5,
):
    """Create an image inpainting function that uses PC samplers.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
      corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for the corrector.
      n_steps: An integer. The number of corrector steps per update of the corrector.
      probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
      continuous: `True` indicates that the score-based model was trained with continuous time.
      denoise: If `True`, add one-step denoising to final samples.
      eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

    Returns:
      An inpainting function.
    """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def pc_inpainter(model, dimg, unet_input,num):
        """Predictor-Corrector (PC) sampler for image inpainting.

        Args:
          model: A score model.
          data: A PyTorch tensor that represents a mini-batch of images to inpaint.
          mask: A 0-1 tensor with the same shape of `data`. Value `1` marks known pixels,
            and value `0` marks pixels that require inpainting.

        Returns:
          Inpainted (complete) images.
        """
        with torch.no_grad():
            timesteps = torch.linspace(sde.T, eps, sde.N)
            x_input = dimg
            x_input = (
                torch.from_numpy(x_input)
                .to(torch.float32)
                .cuda()
                .unsqueeze(0)
                .unsqueeze(0)
            )

         
            x_mean = x_input
            x1 = x_mean
            x2 = x_mean
            x3 = x_mean
            unet_input=torch.tensor(unet_input).to(torch.float32).cuda().unsqueeze(0).unsqueeze(0)
                
            for i in range(sde.N):
                print("===================", i, "===================")
                t = timesteps[i].cuda()
                vec_t = torch.ones(x_input.shape[0], device=t.device) * t
                x, x_mean = predictor_update_fn(
                    x_mean, vec_t, good=unet_input, model=model
                )
      
                x1, x2, x3, x_mean = corrector_update_fn(
                    x1, x2, x3, x_mean, vec_t, good=unet_input, model=model
                )
                x_mean = x_mean.to(torch.float32).cuda()

                x_show = np.array(x_mean.squeeze(0).squeeze(0).cpu())
                mat_dict = {'sensor_data': x_show}  
                #savemat("/home/b/code/LJB/extended BWdata/code/results_xitong_2_6/testpadsecond_" + str(i) + ".mat", mat_dict)
                x_show = np.array(
                    ((x_show - x_show.min()) / (x_show.max() - x_show.min())) 
                )
                cv2.imwrite(
                    "/home/b110/code/LLX/lijiao_1_17/code_1_17/code_1_17/result_2_4/" + str(num)+"/"+str(i) + ".png", x_show * 255.0
                )

    return pc_inpainter
