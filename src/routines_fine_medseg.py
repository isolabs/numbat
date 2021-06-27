import math
import gc
import time
import datetime
import tqdm
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import utils 
import fine_tuners_medseg
import datawork as dw


def train_test_single_epoch(
        epoch_idx, 
        n_epochs, 
        device,
        optimiser,
        scaler,
        mixed_precision,
        model,
        loss_computer,
        train_loader, 
        test_loader,
        learning_rate,
    ):

    """
    Train the provided fine tuning network on labelled data. Generate test accuracy
    """

    # We'll log some stuff
    total_loss_per_epoch = 0

    # Update the learning rate
    for g in optimiser.param_groups:
        g['lr'] = learning_rate

    # Load the batches from the data loader
    for i, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch_idx + 1}/{n_epochs}"):

        # Zero the parameter gradients
        optimiser.zero_grad()

        # Account for mixed precision if using
        with torch.cuda.amp.autocast(enabled=mixed_precision):

            # Get the xs and ys and calculate the loss via MSE
            x = batch['image']
            y = batch['label']
            x = x.to(device)
            y = y.to(device)
            yhat = model(x).to(device)

            # Calculate the task's loss. Input must be N,C..., and target N,...
            loss = loss_computer( yhat, y.long() )
            #loss.requires_grad = True

            # Update the student with mixed precision loss prop
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            # Important to detach so as to not have a memory leak
            total_loss_per_epoch += loss.item()

    # Perform a test run and measure the accuracy

    def calculate_accuracy(loader, desc):
        """ 
        Compare the ground truth and return the test accuracy
        """

        num_true = 0
        # TODO, no hardcode C=4
        per_class_seg_res = [
            {"iou": 0, "dice": 0},
            {"iou": 0, "dice": 0},
            {"iou": 0, "dice": 0},
            {"iou": 0, "dice": 0},
        ]

        num_instances = 0

        # Load the batches from the data loader
        for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader),
                            desc=desc):
            
            # Get the data and compare the prediction to the truth
            x = batch['image']
            y = batch['label']
            x = x.to(device)[0:1]
            y = y.to(device)[0:1]
            yhat = model(x) # B, C~4, W, H

            #print(x.shape)
            #print(y.shape)
            #print(yhat.shape)

            for j in range(len(per_class_seg_res)):

                seg_true = y.detach().cpu().numpy() == j    
                seg_pred = yhat[0,j:j+1,:,:].detach().cpu().numpy()

                def save(tensor, filepath):
                    #print(tensor.shape)
                    tensor = np.einsum("cij->ij", tensor)
                    tensor = np.stack((tensor,)*3, axis=-1)
                    tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))
                    #print(tensor.shape)
                    plt.imsave(filepath, tensor)
                
                save(x.detach().cpu().numpy()[:,0],         f"../logs/seg_in-{i}.png")
                save(y.detach().cpu().numpy(), f"../logs/seg_full-{i}.png")
                #save(yhat[0].detach().cpu().numpy(), f"../logs/seg_pred_full-{i}.png")
                save(seg_pred, f"../logs/seg_pred-{i}-{j}.png")
                #save(seg_true, f"../logs/seg_true-{i}-{j}.png", boolean=True)
                
                _ = utils.evaluate_segmentation(seg_true, seg_pred)
                per_class_seg_res[j]['iou']     += _['iou']
                per_class_seg_res[j]['dice']    += _['dice']

            # Must get the class index with the max value before comparison
            correct = torch.eq( torch.argmax(yhat.detach().cpu(), dim=1), y.detach().cpu() )

            # Aggregate results
            num_true += ( torch.sum(correct).item() / 256**2 )

            # For averaging
            num_instances += len(batch)

            break

        # Average out the IoU and dice
        for i in range(len(per_class_seg_res)):
            per_class_seg_res[i]['iou']     /= num_instances
            per_class_seg_res[i]['dice']    /= num_instances

        # Calculate and return the accuracy
        accuracy = num_true / num_instances * 100
        return accuracy, per_class_seg_res

    # Swap model to evaluation mode before testing, then back again
    model.eval()
    test_accuracy, per_class_seg_res = calculate_accuracy(test_loader, "Testing")
    model.train()

    # Announce
    print("Epoch summary:")
    print(f" - total loss: {total_loss_per_epoch}")
    print(f" - learning rate: {learning_rate}")
    print(f" - test accuracy: {test_accuracy} %")
    for i in range(len(per_class_seg_res)):
        print(f" - class {i} iou: {per_class_seg_res[i]['iou']}")
        print(f" - class {i} dice: {per_class_seg_res[i]['dice']}")

    # Return the metrics
    metrics = {
        "epoch_idx":            epoch_idx,
        "total_loss_per_epoch": total_loss_per_epoch,
        "learning_rate":        learning_rate,
        "test_accuracy":        test_accuracy,
        "per_class_segmentation_results" : per_class_seg_res
    }
    return metrics

def train(fp_config):
    """
    Performs a finetuning experiment based on a provided configuration
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
    # MSL datasets come predefined
    dl_train, dl_test, dl_val = dw.get_decathlon_brains_train_test_val_dataloaders(
        config['fp_data_decathlon_brain_tumor'], 
        config['data_proportion'], 
        config['train_test_val_ratios'],
        config['batch_size'],
        config['perform_shuffle'])
    print("ready")
    
    # ================ LOAD PRETRAINED ================

    print("Loading DINO network ... ", end="")
    
    # Will need to load the saved transformer experiment config
    dino_experiment = torch.load(config['fp_load_dino_model'])
    # Then build the matching architecture
    nn_dino = utils.build_vision_transformer_from_config(dino_experiment['config'])
    # Then load the weights
    nn_dino.load_state_dict(dino_experiment['teacher_state_dict'])
    print("ready")
    
    # ================ WRAP FINETUNE ================

    # Take CLS token and feed into a feedforward NN and output to
    # a classification layer
    print("Wrapping DINO network in classification network ... ", end="")
    fine_tuner = fine_tuners_medseg.FineTunerSegmentation(  vit=nn_dino, 
                                                            vit_embed_dim=dino_experiment['config']['embed_dim'],
                                                            patch_size=dino_experiment['config']['patch_size'],
                                                            n_heads_classes=dino_experiment['config']['n_heads'],
                                                            label_map_size=[256,256], # TODO, make configurable
    )

    fine_tuner.to(device)
    print("ready")

    # Loss is simple here
    loss_computer = nn.CrossEntropyLoss()
    loss_computer.to(device)

    #print(f"Learnable parameters in fine tuner: {sum(p.numel() for p in fine_tuner.parameters() if p.requires_grad)}")
    
    # ================ SCHEDULERS ================
    
    # Create learning rate scheduler and weight decay scheduler for the optimiser
    sch_lr = utils.LinearPiecewiseScheduler(config['lr_values'], config['lr_epochs'])
    
    # ================ OPTIMISER ================
    
    # Preparing the optimiser
    print("Preparing optimiser ... ", end="")
    # Note that it is recommended to only construct this after the 
    # models are on the GPU (if using a GPU)
    optimiser = torch.optim.Adam(fine_tuner.parameters(), 
                                 lr=sch_lr.get_value(0))
    print("ready")
    
    # ================ TRAINING ================
    
    print("Preparing training procedure ... ", end="")

    fp_save = config['fp_save']

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
    print("Next stop segmentation station!")
    
    # Perform as many epochs of training as required
    metrics = []
    n_epochs = config['n_epochs']
    for epoch_idx in range(n_epochs):
        # Get the schedule values
        learning_rate   = sch_lr.get_value(epoch_idx)

        # And train a single epoch
        epoch_metrics = train_test_single_epoch(
            epoch_idx=epoch_idx, 
            n_epochs=n_epochs, 
            device=device,
            scaler=scaler,
            mixed_precision=config['mixed_precision'],
            optimiser=optimiser,
            model=fine_tuner, 
            loss_computer=loss_computer,
            train_loader=dl_test, # We'll actually train with the TEST set here, and test on the VAL set
            test_loader=dl_val,
            learning_rate=learning_rate
        )

        metrics.append(epoch_metrics)
    
        # ================ SAVING ================
        
        print("Saving experiment ... ", end="")
        utils.save_fine_experiment(
            fp_save=fp_save, 
            model=fine_tuner,
            optimiser=optimiser, 
            config=config,
            metrics=metrics
        )
        print("complete")

def inference_segmentation_maps(fp_experiment, fp_out, n_images=4):
    """ 
    Loads an experiment and performs inference to get attention maps
    """
        
    # ================ DEVICE ================
    
    device = utils.report_and_get_device_gpu_preferred()
    
    # ================ LOAD ================

    print("Building architecture and loading weights ... ", end="")

    # Load the experiment at the provided filepath
    experiment = torch.load(fp_experiment)
    experiment_unsupervised         = torch.load(experiment['config']['fp_load_dino_model'])
    experiment_unsupervised_config  = experiment_unsupervised['config']

    # Get everything that we'll need to inference
    config             = experiment['config']
    patch_size         = experiment_unsupervised_config['patch_size']
    metrics            = experiment['metrics']
    model_state_dict   = experiment_unsupervised['teacher_state_dict'] # For vit

    # Rebuild the network
    model = utils.build_vision_transformer_from_config(experiment_unsupervised_config)
    model.load_state_dict(model_state_dict)

    # Then wrap it up
    model = fine_tuners_medseg.FineTunerSegmentation(
        vit=model, 
        vit_embed_dim=experiment_unsupervised_config['embed_dim'],
        patch_size=experiment_unsupervised_config['patch_size'],
        n_heads_classes=experiment_unsupervised_config['n_heads'],
        label_map_size=[256,256],
    )

    # TODO: the teacher always outperforms the student right? Which one
    # to use? Must test

    # Set for inference
    model.eval()
    model.to(device)
    print("ready")
    
    # ================ DATA ================
    
    print("Loading testing data ... ", end="")
    train_loader, test_loader, val_loader = dw.get_decathlon_brains_train_test_val_dataloaders(
        config['fp_data_decathlon_brain_tumor'], 
        config['data_proportion'], 
        config['train_test_val_ratios'],
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
        for i, batch in tqdm.tqdm(enumerate(val_loader), 
                                  desc="Inferencing",
                                  total=n_images):

            # Strict limit
            if i == n_images:
                break

            image = batch['image']
            label = batch['label']
            image = image.to(device)
            label = label.to(device)

            # Setup plotting
            fig, axs = plt.subplots(2, 2)
            fig.set_size_inches(20, 20)
            [axi.set_axis_off() for axi in axs.ravel()]

            # Get the segmentation label
            axs[0,1].imshow(np.einsum("ijk->jki", label.detach().cpu().numpy()), cmap="Reds")

            # Get the segmentation image
            segmentation_map = torch.argmax(model(image), dim=1)
            axs[1,0].imshow(np.einsum("ijk->jki", segmentation_map.detach().cpu().numpy()), cmap="cividis")

            # And save the original image
            image_in = image[0].cpu().numpy()
            image_in = image_in.transpose(1, 2, 0)
            image_in = np.interp(image_in, (image_in.min(), image_in.max()), (0, +1))
            axs[0,0].imshow(image_in, cmap="cividis")
            fig.savefig(
                f"{folderpath}/im-{i}.png", dpi=300)
            plt.close()





    