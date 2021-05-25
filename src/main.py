import torch 
import random 
import numpy as np

import routines

if __name__ == "__main__":

    # This ensures basic program-wide reproducibility , provided
    # no submodule resets the seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # File structure
    fp_repo         = "/home/ubuntu/numbat"
    fp_configs      = f"{fp_repo}/configs"  
    fp_experiments  = f"{fp_repo}/experiments"  
    fp_logs         = f"{fp_repo}/logs"  
    
    # Perform a training experiment under a given experimental configuration,
    # and save it
    
    routines.dino_train(
        f"{fp_configs}/remote-3.yml"
    )
    """

    # Perform inferencing
    routines.inference_attention_maps(
        f"{fp_experiments}/remote-2 2021-05-24 08:30:49.pt",
        f"{fp_logs}",
        n_images=3200
    )
    """
    