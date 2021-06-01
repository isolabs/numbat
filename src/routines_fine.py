import math
import gc
import time
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils 
import fine_tuners
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
        val_loader,
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

            # Calculate the task's loss (must convert to one hot encoding)
            loss = loss_computer( yhat, F.one_hot(y, num_classes=yhat.shape[1]).float() )

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

        # Load the batches from the data loader
        for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader),
                            desc=desc):
            
            # Get the data and compare the prediction to the truth
            x = batch['image']
            y = batch['label']
            x = x.to(device)
            y = y.to(device)

            # Must get the class index with the max value before comparison
            correct = torch.eq( torch.argmax(model(x), dim=1), y )

            # Aggregate results
            num_true += torch.sum(correct).item()

        # Calculate and return the accuracy
        accuracy = num_true / (len(loader.dataset)) * 100
        return accuracy

    # Swap model to evaluation mode before testing, then back again
    model.eval()
    val_accuracy  = calculate_accuracy(val_loader, "Validating")
    test_accuracy = calculate_accuracy(test_loader, "Testing")
    model.train()

    # Announce
    print("Epoch summary:")
    print(f" - total loss: {total_loss_per_epoch}")
    print(f" - learning rate: {learning_rate}")
    print(f" - validation accuracy: {val_accuracy} %")
    print(f" - test accuracy: {test_accuracy} %")

    # Return the metrics
    metrics = {
        "epoch_idx":            epoch_idx,
        "total_loss_per_epoch": total_loss_per_epoch,
        "learning_rate":        learning_rate,
        "val_accuracy":         val_accuracy,
        "test_accuracy":        test_accuracy,
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
    dl_train, dl_test, dl_val = dw.get_msl_train_test_val_dataloaders(config['fp_data_msl'], config['batch_size'])
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
    fine_tuner = fine_tuners.FineTuner( vit=nn_dino, 
                                        vit_embed_dim=dino_experiment['config']['embed_dim'], 
                                        hidden_sizes=config['hidden_sizes'], 
                                        n_classes=config['n_classes'])
    fine_tuner.to(device)
    print("ready")

    # Loss is simple here
    loss_computer = nn.MSELoss()
    loss_computer.to(device)
    
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
    print("'A restless tongue to classify'")
    
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
            train_loader=dl_train, 
            val_loader=dl_val,
            test_loader=dl_test,
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