import yaml
import tqdm

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch

def plot_instance(inst):
    """ 
    Plots an potentially labelled instance
    """
    
    # Require WHC, not CWH
    im = np.einsum('ijk->jki', inst['image'])
    #im = inst['image']
    
    # Make plot
    plt.subplot(111)
    
    # Provide label information if we have it (through the title)
    text = inst['filename']
    if "label" in inst.keys():
        text = f"{text}\n{inst['label_name']} ({inst['label']})"
    plt.title(text)
    
    plt.imshow(im)
    plt.show()

def report_and_get_device_gpu_preferred():
    # Setup devices
    print("Device setup")
    print(f" - CUDA available? {torch.cuda.is_available()}")
    print(f" - CUDA Device count: {torch.cuda.device_count()}")
    print(f" - current CUDA device id: {torch.cuda.current_device()}")
    print(f" - current CUDA device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    print(f" - selected device: {device}")
    return device

def read_yaml(fp):
    # Load the configuration file 
    with open(fp) as f:
        return yaml.load(f, Loader=yaml.FullLoader)