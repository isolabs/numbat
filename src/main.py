import torch 
import random 
import numpy as np

import datawork as dw

import routines_dino
import routines_dino_medseg
import routines_fine
import routines_fine_medseg

if __name__ == "__main__":

    # This ensures basic program-wide reproducibility , provided
    # no submodule resets the seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # File structure, change as necessary for your machine
    fp_repo         = "/home/prang/dev/numbat"
    fp_data         = f"{fp_repo}/data"  
    fp_configs      = f"{fp_repo}/configs"  
    fp_experiments  = f"{fp_repo}/experiments"  
    fp_logs         = f"{fp_repo}/logs"  
    
    """
    # Perform a dino training experiment under a given experimental configuration,
    # and save it
    routines_dino.train(
        f"{fp_configs}/remote-3.yml"
    )
    # Perform inferencing
    routines_dino.inference_attention_maps(
        f"{fp_experiments}/remote-2 2021-05-24 08:30:49.pt",
        f"{fp_logs}",
        n_images=3200
    )

    # Perform a fine tuning experiment
    routines_fine.train(f"{fp_configs}/fine-4.yml")
    """

    # MED SEG
    # Perform unsupervised training
    if False:
        routines_dino_medseg.train(
            f"{fp_configs}/medseg-dino-6-no-dino.yml"
        )
    
    # Perform inferencing
    if False:
        routines_dino_medseg.inference_attention_maps(
            f"{fp_experiments}/medseg-dino-5 2021-06-26 10:52:52.pt",
            f"{fp_logs}",
            n_images=128
        )

    # Perform a fine tuning experiment
    if True:
        routines_fine_medseg.train(f"{fp_configs}/medseg-fine-1.yml")

    # Perform segmentation inferencing
    if False:
        routines_fine_medseg.inference_segmentation_maps(
            f"{fp_experiments}/medseg-fine-2-no-dino 2021-06-27 02:57:58.pt",
            f"{fp_logs}",
            n_images=128
        )