import math
import gc
import time
import tqdm


def train_single_epoch(
        epoch_idx, 
        n_epochs, 
        device,
        optimiser,
        scaler,
        mixed_precision,
        train_loader, 
        learning_rate,
    ):
    """
    Train the provided networks in the DINO framework for a single epoch. Returns
    the training metrics
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

            # Calculate the task's loss 
            loss = 99

            # Update the DINO loss computer's centering parameter
            loss_computer.update_center(out_teacher, cent_rate_m)

            # Update the student with mixed precision loss prop
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            # Important to detach so as to not have a memory leak
            total_loss_per_epoch += loss.item()

    # Announce
    print("Epoch summary:")
    print(f" - total loss: {total_loss_per_epoch}")
    print(f" - learning rate: {learning_rate}")

    # Return the metrics
    metrics = {
        "epoch_idx":            epoch_idx,
        "total_loss_per_epoch": total_loss_per_epoch,
        "learning_rate":        learning_rate,
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
    # MSL
    # TODO
    print("ready")
    
    # ================ LOAD PRETRAINED ================
    
    # TODO: will need to load the saved transformer experiment config

    print("Loading DINO network ... ", end="")
    nn_student = utils.build_vision_transformer_from_config(fp_config_dino)
    print("ready")
    
    # ================ WRAP FINETUNE ================

    # TODO: take CLS token and feed into a feedforward NN and output to
    # a classification layer
    
    print("Wrapping DINO network in classification network ... ", end="")
    
    print("ready")
    
    # ================ SCHEDULERS ================
    
    # Create learning rate scheduler and weight decay scheduler for the optimiser
    sch_lr = utils.LinearPiecewiseScheduler(config['lr_values'], config['lr_epochs'])
    
    # ================ OPTIMISER ================
    
    # Preparing the optimiser
    print("Preparing optimiser ... ", end="")
    # Note that it is recommended to only construct this after the 
    # models are on the GPU (if using a GPU)
    optimiser = torch.optim.Adam(nn_student.parameters(), 
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
    
    # Perform as many epochs of training as required
    metrics = []
    n_epochs = config['n_epochs']
    for epoch_idx in range(n_epochs):
        # Get the schedule values
        learning_rate   = sch_lr.get_value(epoch_idx)

        # And train a single epoch
        epoch_metrics = train_single_epoch(
            epoch_idx=epoch_idx, 
            n_epochs=n_epochs, 
            device=device,
            scaler=scaler,
            mixed_precision=config['mixed_precision'],
            optimiser=optimiser,
            model=model, 
            train_loader=train_loader, 
            learning_rate=learning_rate
        )

        metrics.append(epoch_metrics)
    
        # ================ SAVING ================
        
        print("Saving experiment ... ", end="")
        utils.save_fine_experiment(
            fp_save=fp_save, 
            model=model,
            optimiser=optimiser, 
            loss_computer=loss_computer, 
            config=config,
            metrics=metrics
        )
        print("complete")