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

class LinearPiecewiseScheduler():
    """ 
    A scheduler that linearly interpolates between the desired values
    over a course of steps
    """
    def __init__(self, defined_values, defined_steps):
        """ 
        Precalculate the values at every step based on a list of values
        at certain steps, using linear piecewise interpolation
        """

        if len(defined_values) != len(defined_steps):
            raise ValueError("There must be a provided step for each value. \
                              Ensure that the defined values and step lists \
                              are the same length.")

        # Perform precalculation
        l=[]
        for i in range(len(defined_values) - 1):
            # Get references to all the items that we'll need to
            # perform the interpolation
            value = defined_values[i]
            step  = defined_steps[i]
            next_value = defined_values[i+1]
            next_step  = defined_steps[i+1]

            values = np.linspace(value, next_value, 
                                 num=(next_step - step),
                                 endpoint=False)
            l.extend(list(values))
        l.append(defined_values[-1])
        
        # Save the list of values
        self.values_at_steps = l

    def get_value(self, step):
        """ 
        Return the value at the provided step. If before the range return
        the first value, if beyond the range return the last value
        """
        step = min(step, len(self.values_at_steps) - 1)
        step = max(step, 0)
        return self.values_at_steps[step]