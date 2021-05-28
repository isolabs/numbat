import math
import gc
import time
import datetime
import tqdm
import os

import torch
import torch.nn.functional as F

import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np

import utils
import datawork as dw
import modules.vit as vit
from modules.multicropper import MultiCropper
from modules.loss import LossComputerDINO

def train_single_epoch(
        epoch_idx, 
        n_epochs, 
        device,
        optimiser,
        scaler,
        mixed_precision,
        nn_student,
        nn_teacher,
        loss_computer,
        train_loader, 
        multicropper,
        learning_rate,
        #weight_decay,
        temp_student,
        temp_teacher,
        cent_rate_m,
        lambda_ema
    ):
    """
    Train the provided networks in the DINO framework for a single epoch. Returns
    the training metrics
    """

    # We'll log some stuff
    total_loss_per_epoch = 0

    # Update the learning rate
    for g in optimiser.param_groups:
        g['lr']           = learning_rate
        #g['weight_decay'] = weight_decay

    n_batches = 0

    # Load the batches from the data loader
    for i, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch_idx + 1}/{n_epochs}"):

        # Zero the parameter gradients
        optimiser.zero_grad()

        # Account for mixed precision if using
        with torch.cuda.amp.autocast(enabled=mixed_precision):

            # Get the global and local views from the images
            global_crops, local_crops = multicropper.crop(batch['image'])

            # Move them to the desired device
            # Note that the data version of the 'to' function is not 
            # an in place operation
            global_crops = global_crops.to(device)
            local_crops = local_crops.to(device)

            # Push the global views through the teacher network
            out_teacher = nn_teacher(global_crops)

            # Push all the views through the student network
            out_student_global = nn_student(global_crops)
            out_student_local  = nn_student(local_crops)

            # Calculate the DINO loss 
            loss =  loss_computer(
                        out_student_global, 
                        out_student_local, 
                        out_teacher,
                        temp_student,
                        temp_teacher,
                        cent_rate_m
                    )

            # Update the DINO loss computer's centering parameter
            loss_computer.update_center(out_teacher, cent_rate_m)

            # Update the student with mixed precision loss prop
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            # Important to detach so as to not have a memory leak
            total_loss_per_epoch += loss.item()
            n_batches += 1

    # Use exponential moving average (EMA, or momentum encoder) of the student 
    # weights to update the teacher. Importantly - do not perform these calculations
    # with the intention to calculation gradients (its a waste)
    with torch.no_grad():
        for param_q, param_k in zip(nn_student.parameters(), nn_teacher.parameters()):
            param_k.data.mul_(lambda_ema).add_((1 - lambda_ema) * param_q.detach().data)

    total_loss_per_epoch /= n_batches

    # Announce
    print("Epoch summary:")
    print(f" - loss per batch: {total_loss_per_epoch}")
    print(f" - learning rate: {learning_rate}")
    #print(f" - weight decay: {weight_decay}")
    print(f" - student temperature: {temp_student}")
    print(f" - teacher temperature: {temp_teacher}")
    print(f" - centering rate parameter: {cent_rate_m}")
    print(f" - lambda teacher update proportion: {lambda_ema}")

    # Return the metrics
    metrics = {
        "epoch_idx":            epoch_idx,
        "total_loss_per_epoch": total_loss_per_epoch,
        "learning_rate":        learning_rate,
        #"weight_decay":         weight_decay,
        "temp_student":         temp_student,
        "temp_teacher":         temp_teacher,
        "cent_rate_m":          cent_rate_m,
        "lambda_ema":           lambda_ema,
    }
    return metrics

def dino_train(fp_config, fp_save):
    """
    Performs a training experiment based on a provided configuration
    """
    
    # ================ CONFIGURATION ================
    
    # Load the configuration file 
    config = utils.read_yaml(fp_config)

    # Announce the configuration
    print(f"Using the configuration at: {fp_config}")
    for k in config.keys(): print(f" - {k}: {config[k]}")
        
    # ================ DEVICE ================
    
    device = utils.report_and_get_device_gpu_preferred()
    
    # ================ DATA ================
    
    print("Loading training data ... ", end="")
    train_loader, test_loader = dw.get_mars32k_train_test_dataloaders(
        config['fp_data_mars32k'], 
        config['data_proportion'], 
        config['train_test_ratio'],
        config['batch_size'],
        config['perform_shuffle'])
    print("ready")
    
    # ================ STUDENT AND TEACHER NETWORKS ================
    
    # Create the student network
    print("Building student network ... ", end="")
    nn_student = utils.build_vision_transformer_from_config(config)
    print("ready")
    
    # Create the teacher network
    print("Building teacher network ... ", end="")
    nn_teacher = utils.build_vision_transformer_from_config(config)
    
    # Copy the student weights over to the teacher (initially they are the same)
    nn_teacher.load_state_dict(nn_student.state_dict())
    
    # There is no backpropagation through the teacher
    for param in nn_teacher.parameters():
        param.requires_grad = False

    # Move them both to the desired device
    nn_student.to(device)
    nn_teacher.to(device)

    print("ready")
    
    # ================ MULTICROPPER ================
    
    # Build the multicropper module
    print("Instantiating multicropper ... ", end="")
    multicropper = MultiCropper(
                        global_size=config['global_size'],
                        local_size=config['local_size'],
                        n_global_crops=config['n_global_crops'],
                        n_local_crops=config['n_local_crops'])
    print("ready")
    
    # ================ SCHEDULERS ================
    
    # Create learning rate scheduler and weight decay scheduler for the optimiser
    sch_lr = utils.LinearPiecewiseScheduler(config['lr_values'], config['lr_epochs'])
    #sch_wd = utils.LinearPiecewiseScheduler(config['weight_decay_values'], 
    #                                        config['weight_decay_epochs'])
    
    # Create temperature schedulers
    sch_temp_student = utils.LinearPiecewiseScheduler(config['temp_student_values'], 
                                                      config['temp_student_epochs'])

    # Too high temperature at the start may cause the teacher's training
    # to be unstable. Investigate the use of warm up as necessary
    sch_temp_teacher = utils.LinearPiecewiseScheduler(config['temp_teacher_values'], 
                                                      config['temp_teacher_epochs'])

    # The centering rate parameter (usually in range [0.9, 0.999])
    sch_cent_rate_m = utils.LinearPiecewiseScheduler(config['cent_rate_m_values'], 
                                                     config['cent_rate_m_epochs'])

    # The lambda weight transfer parameter (the amount of teacher network v. student
    # network to include in the next iteration of the teacher network)
    # TODO a cosine scheduler should be used if possible
    sch_lambda_ema = utils.LinearPiecewiseScheduler(config['lambda_ema_values'], 
                                                       config['lambda_ema_epochs'])
    
    # ================ OPTIMISER ================
    
    # Preparing the optimiser
    print("Preparing optimiser ... ", end="")
    # Note that it is recommended to only construct this after the 
    # models are on the GPU (if using a GPU)
    optimiser = torch.optim.Adam(nn_student.parameters(), 
                                  lr=sch_lr.get_value(0))
    print("ready")
    
    # ================ DINO LOSS ================
    
    # Prepare the loss module
    print("Preparing DINO loss module ... ", end="")
    loss_computer = LossComputerDINO(output_dim=config['embed_dim'])
    loss_computer.to(device)
    print("ready")    
    
    # ================ TRAINING ================
    
    print("Preparing training procedure ... ", end="")

    # Create a scalar for mixed precision as 
    # desired (no op if flag is false) 
    scaler = torch.cuda.amp.GradScaler(enabled=config['mixed_precision'])

    # Synchronise
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    print("ready")
        
    time.sleep(0.5)
    
    # Perform as many epochs of training as required
    metrics = []
    n_epochs = config['n_epochs']
    for epoch_idx in range(n_epochs):
        # Get the schedule values
        learning_rate   = sch_lr.get_value(epoch_idx)
        #weight_decay    = sch_wd.get_value(epoch_idx)
        temp_student    = sch_temp_student.get_value(epoch_idx)
        temp_teacher    = sch_temp_teacher.get_value(epoch_idx)
        cent_rate_m     = sch_cent_rate_m.get_value(epoch_idx)
        lambda_ema      = sch_lambda_ema.get_value(epoch_idx)

        # And train a single epoch
        epoch_metrics = train_single_epoch(
            epoch_idx=epoch_idx, 
            n_epochs=n_epochs, 
            device=device,
            scaler=scaler,
            mixed_precision=config['mixed_precision'],
            optimiser=optimiser,
            nn_student=nn_student,
            nn_teacher=nn_teacher,
            loss_computer=loss_computer,
            train_loader=train_loader, 
            multicropper=multicropper,
            learning_rate=learning_rate,
            #weight_decay=weight_decay,
            temp_student=temp_student,
            temp_teacher=temp_teacher,
            cent_rate_m=cent_rate_m,
            lambda_ema=lambda_ema
        )

        metrics.append(epoch_metrics)
    
        # ================ SAVING ================
        
        print("Saving experiment ... ", end="")
        utils.save_experiment(
            fp_save=fp_save, 
            nn_student=nn_student, 
            nn_teacher=nn_teacher, 
            optimiser=optimiser, 
            loss_computer=loss_computer, 
            config=config,
            metrics=metrics
        )
        print("complete")
    
def inference_attention_maps(fp_experiment, fp_out, n_images=4):
    """ 
    Loads an experiment and performs inference to get attention maps
    """
        
    # ================ DEVICE ================
    
    device = utils.report_and_get_device_gpu_preferred()
    
    # ================ LOAD ================

    print("Building architecture and loading weights ... ", end="")

    # Load the experiment at the provided filepath
    experiment = torch.load(fp_experiment)

    # Get everything that we'll need to inference
    config             = experiment['config']
    patch_size         = config['patch_size']
    metrics            = experiment['metrics']
    student_state_dict = experiment['student_state_dict']
    teacher_state_dict = experiment['teacher_state_dict']

    # Rebuild the network
    model = utils.build_vision_transformer_from_config(config)
    model.load_state_dict(student_state_dict)

    # TODO: the teacher always outperforms the student right? Which one
    # to use? Must test

    # Set for inference
    model.eval()
    model.to(device)
    print("ready")
    
    # ================ DATA ================
    
    print("Loading testing data ... ", end="")
    train_loader, test_loader = dw.get_mars32k_train_test_dataloaders(
        config['fp_data_mars32k'], 
        config['data_proportion'], 
        config['train_test_ratio'],
        1, # We'll inference one image at a time for simplicity
        config['perform_shuffle'])
    print("ready")
    
    # ================ INFERENCE ================

    print("Preparing inference procedure ... ", end="")

    # Synchronise
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()

    # We'll save the inference results to a folder
    # Append a timestamp (milliseconds not required) and file extension 
    # to the filepath
    timestamp = datetime.datetime.now().replace(microsecond=0)
    folderpath = f"{fp_out}/inference {n_images}-ims {timestamp}"
    os.mkdir(folderpath)

    print("ready")
        
    time.sleep(0.5)

    # Don't waste computation
    with torch.no_grad():

        # Only do a set number of images
        for i, batch in tqdm.tqdm(enumerate(test_loader), 
                                  desc="Inferencing",
                                  total=n_images):

            # Strict limit
            if i == n_images:
                break

            # We can feed in any image as long as its divisible by the patch
            # size, so lets make this divisible by the patch size. Shape is
            # currently (B=1, C, W, H)
            image = batch['image']
            new_w = image.shape[2] - image.shape[2] % patch_size
            new_h = image.shape[3] - image.shape[3] % patch_size
            image = image[:, :, :new_w, :new_h]

            # The output map will be this big
            w_map = image.shape[2] // patch_size
            h_map = image.shape[3] // patch_size

            # Put it on the desired device
            image = image.to(device)

            # Get the output, shape is (B=1, n_heads, X, X)
            attn = model(image, return_only_last_attn=True)
            n_heads = attn.shape[1]

            # We only want the CLS token SELF attention, so not the heads
            # attention on itself in the last dim, and only the first dim on
            # the second to last dim
            attn = attn[0, :, 0, 1:]                    # (B=1, n_heads, 1, X - 1)
            attn = attn.reshape(n_heads, w_map, h_map)  # (NH, WMAP, HMAP)
            attn = attn.unsqueeze(0)                    # (B=1, NH, WMAP, HMAP)

            # Scale it up to an image and move the final maps to the cpu as 
            # numpy matrices so they can be plotted easily
            attn_map_scaled = F.interpolate(
                attn,
                scale_factor=patch_size,
                mode="nearest"
            ).cpu().numpy() # (B=1, NH, ~W, ~H)
            attn_map_scaled = attn_map_scaled[0] # (NH, ~W, ~H)

            # Average across the heads to get a 2D map (~W, ~H)
            attn_map_2d = np.mean(attn_map_scaled, axis=0)
            
            # Save the images to the provided filepath in a folder
            plt.imsave(
                f"{folderpath}/inf-{i}.png", 
                arr=attn_map_2d,
                cmap="bone",
                format="png")

            image_in = image[0].cpu().numpy()
            image_in = image_in.transpose(1, 2, 0)
            image_in = np.interp(image_in, (image_in.min(), image_in.max()), (0, +1))
            plt.imsave(
                f"{folderpath}/im-{i}.png", 
                arr=image_in,
                format="png")





    