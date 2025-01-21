# Standard library
from argparse import ArgumentParser
import os
import torch
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap, transformer_auto_wrap_policy
from torch.cuda.amp.grad_scaler import GradScaler
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   CheckpointImpl,
   apply_activation_checkpointing,
)
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
import sys
import random
import time
import numpy as np

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS
)
from timm.models.vision_transformer import Block
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock
)





def replace_constant(y, yhat, out_variables):
    for i in range(yhat.shape[1]):
        # if constant replace with ground-truth value
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = y[:, i]
    return yhat


def training_step(
    batch,
    batch_idx,
    net,
    device: int,
    train_loss_metric) -> torch.Tensor:
    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)
        
    yhat = net.forward(x)
    yhat = replace_constant(y, yhat, out_variables)
    losses = train_loss_metric(yhat, y)
    loss_name = getattr(train_loss_metric, "name", "loss")
    if losses.dim() == 0:  # aggregate loss only
        loss = losses
    else:  # per channel + aggregate
        loss = losses[-1]
        
    return loss


def validation_step(
    batch,
    batch_idx: int,
    net,
    device: int,
    val_loss_metrics,
    val_target_transforms) -> torch.Tensor:

    return evaluate_func(batch, "val",net,device,val_loss_metrics,val_target_transforms)


def evaluate_func(
    batch,
    stage: str,
    net,
    device: int,
    loss_metrics,
    target_transforms):

    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)
 
    yhat = net.forward(x)
    yhat = replace_constant(y, yhat, out_variables)

    if stage == "val":
        loss_fns = loss_metrics
        transforms = target_transforms
    elif stage == "test":
        loss_fns = loss_metrics
        transforms = self.target_transforms
    else:
        raise RuntimeError("Invalid evaluation stage")
    loss_dict = {}
    for i, lf in enumerate(loss_fns):

        if transforms is not None and transforms[i] is not None:
            yhat_ = transforms[i](yhat)
            y_ = transforms[i](y)
        losses = lf(yhat_, y_)
        loss_name = getattr(lf, "name", f"loss_{i}")
        if losses.dim() == 0:  # aggregate loss
            loss_dict[f"{stage}/{loss_name}:agggregate"] = losses
        else:  # per channel + aggregate
            for var_name, loss in zip(out_variables, losses):
                name = f"{stage}/{loss_name}:{var_name}"
                loss_dict[name] = loss
            loss_dict[f"{stage}/{loss_name}:aggregate"] = losses[-1]
    return loss_dict




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(device):

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = dist.get_rank()
    local_rank = int(os.environ['SLURM_LOCALID'])


    print("world_size",world_size,"world_rank",world_rank,"local_rank",local_rank,flush=True)


    parser = ArgumentParser()
    parser.add_argument("era5_low_res_dir")
    parser.add_argument("era5_high_res_dir")
    parser.add_argument("preset", choices=["resnet", "unet", "vit","res_slimvit"])
    parser.add_argument(
        "variable", choices=["t2m", "z500", "t850","u10"], help="The variable to predict."
    )
    parser.add_argument("--summary_depth", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()


    if world_rank==0:
        print("args is",args,flush=True)

    # Set up data
    variables = [
        "land_sea_mask",
        "orography",
        "lattitude",
    #    "toa_incident_solar_radiation",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential",
        "temperature",
    #    "relative_humidity",
        "specific_humidity",
    #    "u_component_of_wind",
    #    "v_component_of_wind",
    ]
    out_var_dict = {
        "t2m": "2m_temperature",
#        "z500": "geopotential_500",
#        "t850": "temperature_850",
#        "u10": "10m_u_component_of_wind"
    }
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)
    

    if world_rank==0:
        print("in_vars",in_vars,flush=True)

    #load data module
    data_module = cl.data.IterDataModule(
        "downscaling",
        args.era5_low_res_dir,
        args.era5_high_res_dir,
        in_vars,
        out_vars=[out_var_dict[args.variable]],
        subsample=1,
        batch_size=32,
        buffer_size=500,
        num_workers=1,
    ).to(device)
    data_module.setup()

    # Set up deep learning model
    model, train_loss,val_losses,test_losses,train_transform,val_transforms,test_transforms = cl.load_downscaling_module(device,data_module=data_module, architecture=args.preset)
  
    if dist.get_rank()==0:
        print("train_loss",train_loss,"train_transform",train_transform,"val_losses",val_losses,"val_transforms",val_transforms,flush=True)
 
    model = model.to(device)


    #load model checkpoint

    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            print("model resume from checkpoint",args.checkpoint," Checkpoint path found.",flush=True)

            #map_location = 'cuda:'+str(device)
            map_location = 'cpu'

            checkpoint = torch.load(args.checkpoint,map_location=map_location)
            model.load_state_dict(checkpoint['model_state_dict'])

            del checkpoint
        else:
            print("resume from checkpoint was set to True. But the checkpoint path does not exist.",flush=True)

            sys.exit("checkpoint path does not exist")





    seed_everything(0)
    default_root_dir = f"{args.preset}_downscaling_{args.variable}"


    #set up layer wrapping
    if args.preset =="vit" or args.preset=="res_slimvit":
   
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Block # < ---- Your Transformer layer class
            },
        )

        check_fn = lambda submodule: isinstance(submodule, Block)



    elif args.preset =="unet":
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                UpBlock,DownBlock,MiddleBlock  # < ---- Your Transformer layer class
            },
        )

        check_fn = lambda submodule: isinstance(submodule, UpBlock) or isinstance(submodule, DownBlock) or isinstance(submodule, MiddleBlock)



    else: #resnet
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                ResidualBlock  # < ---- Your Transformer layer class
            },
        )
        check_fn = lambda submodule: isinstance(submodule, ResidualBlock)



    #bfloat16 policy
    bfloatPolicy = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )



    #fully sharded FSDP

    model = FSDP(model, device_id = local_rank, process_group= None,sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD,auto_wrap_policy = auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False)



    #model = DDP(model, device_ids=[local_rank], output_device=[local_rank]) 

    #activation checkpointing
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
    )


    if torch.distributed.get_rank()==0:
        print("model is ",model,flush=True)




    #load optimzier and scheduler


    optimizer = cl.load_optimizer(
	model, "adamw", {"lr": 1e-4, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    )

    scheduler = cl.load_lr_scheduler(
	"linear-warmup-cosine-annealing",
	optimizer,
	{
	    "warmup_epochs": 2,  
	    "max_epochs": 50,
	    "warmup_start_lr": 1e-8,
	    "eta_min": 1e-8,
	},
    )


    epoch_start = 0
    if args.checkpoint is not None:

        print("optimizer resume from checkpoint",args.checkpoint," Checkpoint path found.",flush=True)

        #map_location = 'cuda:'+str(device)
        map_location = 'cpu'

        checkpoint = torch.load(args.checkpoint,map_location=map_location)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']+1

        del checkpoint



    #get latitude and longitude
    lat, lon = data_module.get_lat_lon()

    # get train data loader
    train_dataloader = data_module.train_dataloader()

    # get validation data loader
    val_dataloader = data_module.val_dataloader()



    #perform training
    for epoch in range(epoch_start,args.max_epochs):
    
        #tell the model that we are in train mode. Matters because we have the dropout
        model.train()
        loss = 0.0
        epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
        if world_rank==0:
            print("epoch ",epoch,flush=True)
    
        for batch_idx, batch in enumerate(train_dataloader):

            if world_rank==0:
                torch.cuda.synchronize(device=device)
                tic1 = time.perf_counter() 

            loss = training_step(batch, batch_idx,model,device,train_loss)

            epoch_loss += loss.detach()
    
            if world_rank==0:
                print("epoch: ",epoch,"batch_idx",batch_idx,"world_rank",world_rank," loss ",loss,flush=True)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    
            if world_rank==0:
                print("rank",world_rank,"batch_idx",batch_idx,"get_lr ",scheduler.get_lr(),"after optimizer step torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)


            if world_rank==0:
                torch.cuda.synchronize(device=device)
                tic4 = time.perf_counter() 
                print(f"my rank {dist.get_rank()}. tic4-tic1 in {(tic4-tic1):0.4f} seconds\n",flush=True)
    

        scheduler.step()
    
        if world_rank==0:
            print("epoch: ",epoch," epoch_loss ",epoch_loss,flush=True)



        if world_rank ==0:    
            checkpoint_path = "/lustre/orion/nro108/scratch/xf9/checkpoints/climate" 
            # Check whether the specified checkpointing path exists or not
            isExist = os.path.exists(checkpoint_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(checkpoint_path)
                print("The new checkpoint directory is created!")        


        print("rank",world_rank,"Before torch.save torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)


        model_states = model.state_dict()
        optimizer_states = optimizer.state_dict()
        scheduler_states = scheduler.state_dict()

        if world_rank == 0 :
     
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_states,
                'optimizer_state_dict': optimizer_states,
                'scheduler_state_dict': scheduler_states,
                }, checkpoint_path+"/"+"ERA5"+"_rank_"+str(world_rank)+"_epoch_"+ str(epoch) +".ckpt")
     
        print("rank",world_rank,"After torch.save torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)

        dist.barrier()
        del model_states
        del optimizer_states
        del scheduler_states


        #perform validation
        if epoch%2==0:
            with torch.no_grad():
                #tell the model that we are in eval mode. Matters because we have the dropout
                model.eval()

                if world_rank==0:
                    print("val epoch ",epoch,flush=True)
    
                for batch_idx, batch in enumerate(val_dataloader):

                    if world_rank==0:
                        torch.cuda.synchronize(device=device)
                        tic1 = time.perf_counter() 

                    losses = validation_step(batch, batch_idx,model,device,val_losses,val_transforms)
    
                    if world_rank==0:
                        print("val epoch: ",epoch,"batch_idx",batch_idx,"world_rank",world_rank," losses ",losses,flush=True)
                    


if __name__ == "__main__":

    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()


    dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)


    print("Using dist.init_process_group. world_size ",world_size,flush=True)
    
    main(device)

    dist.destroy_process_group()
