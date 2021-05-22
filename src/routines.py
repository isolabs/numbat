import math
import gc
import time
import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

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
        temp_student,
        temp_teacher,
        cent_rate_m,
        lambda_ema
    ):
    """
    Train the provided networks in the DINO framework for a single epoch
    """

    # We'll log some stuff
    total_loss_per_epoch = 0

    # Load the batches from the data loader
    for i, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch_idx + 1}/{n_epochs}"):

        # Zero the parameter gradients
        optimiser.zero_grad()

        # Update the learning rate
        for g in optimiser.param_groups:
            g['lr'] = learning_rate

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
            total_loss_per_epoch += loss

            # Update the DINO loss computer's centering parameter
            loss_computer.update_center(out_teacher, cent_rate_m)

        # Update the student with mixed precision loss prop
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

    # Keep a running track of the loss
    total_loss_per_epoch += loss

    # Announce
    print("Epoch summary:")
    print(f" - total loss: {total_loss_per_epoch}")
    print(f" - learning rate: {learning_rate}")
    print(f" - student temperature: {temp_student}")
    print(f" - teacher temperature: {temp_teacher}")
    print(f" - centering rate parameter: {cent_rate_m}")
    print(f" - lambda teacher update proportion: {lambda_ema}")

    # Use exponential moving average (EMA, or momentum encoder) of the student 
    # weights to update the teacher. Importantly - do not perform these calculations
    # with the intention to calculation gradients (its a waste)
    with torch.no_grad():
        for param_q, param_k in zip(nn_student.parameters(), nn_teacher.parameters()):
            param_k.data.mul_(lambda_ema).add_((1 - lambda_ema) * param_q.detach().data)

def dino_train(fp_config):
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
    
    # This is our simple preprocessing chain
    transform = transforms.Compose([
        dw.TransformToFloat(),
        dw.TransformNormalizeMars32k()
    ])
    # Apply these transformations and load
    dataset = dw.DatasetMars32k(config['fp_data_mars32k'], transform=transform)

    # Downsize the amount of data being used if appropriate
    data_downsize = math.floor( len(dataset) * config['data_proportion'] )
    dataset = torch.utils.data.Subset(dataset, range(data_downsize))
    
    # Make the split of data
    n_train = math.floor( len(dataset) * config['train_test_ratio'] )
    n_test  = ( len(dataset) - n_train )
    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    # Convert to dataloaders
    batch_size = config['batch_size']
    shuffle = config['shuffle_data']
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=shuffle)
    print("ready")
    
    # ================ STUDENT AND TEACHER NETWORKS ================
        
    # This helper function builds a vision transformer from the config file
    def build_vision_transformer():
        return vit.VisionTransformer(
            global_size=config['global_size'],
            n_classes=config['n_classes'],
            in_channels=config['in_channels'],
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim'],
            n_blocks=config['n_blocks'],
            n_heads=config['n_heads'],
            qkv_drop_p=config['qkv_drop_p'], 
            embed_drop_p=config['embed_drop_p'],
            mlp_hidden_ratio=config['mlp_hidden_ratio'],
            mlp_drop_p=config['mlp_drop_p'])
    
    # Create the student network
    print("Building student network ... ", end="")
    nn_student = build_vision_transformer()
    print("ready")
    
    # Create the teacher network
    print("Building teacher network ... ", end="")
    nn_teacher = build_vision_transformer()
    
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
    
    # Create learning rate schedulers
    sch_lr = utils.LinearPiecewiseScheduler(config['lr_values'], config['lr_epochs'])
    
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
    loss_computer = LossComputerDINO(output_dim=config['n_classes'])
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

    print("This is where the fun begins!")
    
    # Perform as many epochs of training as required
    n_epochs = config['n_epochs']
    for epoch_idx in range(n_epochs):
        # Get the schedule values
        learning_rate   = sch_lr.get_value(epoch_idx)
        temp_student    = sch_temp_student.get_value(epoch_idx)
        temp_teacher    = sch_temp_teacher.get_value(epoch_idx)
        cent_rate_m     = sch_cent_rate_m.get_value(epoch_idx)
        lambda_ema      = sch_lambda_ema.get_value(epoch_idx)

        # And train a single epoch
        train_single_epoch(
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
            temp_student=temp_student,
            temp_teacher=temp_teacher,
            cent_rate_m=cent_rate_m,
            lambda_ema=lambda_ema)
    