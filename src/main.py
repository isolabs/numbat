import torch 
import random 
import numpy as np

import routines_dino
import routines_fine

if __name__ == "__main__":

    # This ensures basic program-wide reproducibility , provided
    # no submodule resets the seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # File structure
    fp_repo         = "/home/prang/dev/numbat"
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
    """

    # Perform a fine tuning experiment
    routines_fine.train(f"{fp_configs}/fine-4.yml")
    