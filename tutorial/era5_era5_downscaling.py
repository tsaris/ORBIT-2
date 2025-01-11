# Standard library
from argparse import ArgumentParser
import os
import torch
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap, transformer_auto_wrap_policy
from torch.cuda.amp.grad_scaler import GradScaler
import torch.distributed as dist
from datetime import timedelta
import sys
import random
import numpy as np
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   CheckpointImpl,
   apply_activation_checkpointing,
)
from torch.distributed.fsdp import MixedPrecision

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)
from timm.models.vision_transformer import Block
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock
)


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


    if world_rank==0:
        print("world_size",world_size,"world_rank",world_rank,flush=True)


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


    dm = cl.data.IterDataModule(
        "downscaling",
        args.era5_low_res_dir,
        args.era5_high_res_dir,
        in_vars,
        out_vars=[out_var_dict[args.variable]],
        subsample=1,
        batch_size=32,
        buffer_size=500,
        num_workers=1,
    )
    dm.setup()

    # Set up deep learning model
    model = cl.load_downscaling_module(device,data_module=dm, architecture=args.preset)

    seed_everything(0)
    default_root_dir = f"{args.preset}_downscaling_{args.variable}"

    if args.preset =="vit" or args.preset=="res_slimvit":
   
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Block  # < ---- Your Transformer layer class
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
    model = FSDP(model, device_id=local_rank, process_group= None, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, auto_wrap_policy = auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )

    #activation checkpointing
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
    )


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
