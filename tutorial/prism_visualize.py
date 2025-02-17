import climate_learn as cl
import torch
import os
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
import yaml

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


os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
os.environ['MASTER_PORT'] = "29500"
os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
os.environ['RANK'] = os.environ['SLURM_PROCID']

world_size = int(os.environ['SLURM_NTASKS'])
world_rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])


torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()

torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

# assuming we are downscaling geopotential from ERA5

config_path = sys.argv[1]

if world_rank==0:
    print("config_path",config_path,flush=True)

conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

max_epochs=conf['trainer']['max_epochs']
checkpoint_path = conf['trainer']['checkpoint']
batch_size = conf['trainer']['batch_size']
num_workers = conf['trainer']['num_workers']
buffer_size = conf['trainer']['buffer_size']

pretrain_path = conf['trainer']['pretrain']
low_res_dir = conf['data']['low_res_dir']
high_res_dir = conf['data']['high_res_dir']
preset = conf['model']['preset']
out_variable = conf['data']['out_variable']
dict_in_variables = conf['data']['dict_in_variables']
out_var_dict = conf['data']['out_var_dict']

lr = float(conf['model']['lr'])
beta_1 = float(conf['model']['beta_1'])
beta_2 = float(conf['model']['beta_2'])
weight_decay = float(conf['model']['weight_decay'])
warmup_epochs =  conf['model']['warmup_epochs']
warmup_start_lr =  float(conf['model']['warmup_start_lr'])
eta_min =  float(conf['model']['eta_min'])

superres_mag = conf['model']['superres_mag']
cnn_ratio = conf['model']['cnn_ratio']
patch_size =  conf['model']['patch_size']
embed_dim = conf['model']['embed_dim']
depth = conf['model']['depth']
decoder_depth = conf['model']['decoder_depth']
num_heads = conf['model']['num_heads']
mlp_ratio = conf['model']['mlp_ratio']
drop_path = conf['model']['drop_path']
drop_rate = conf['model']['drop_rate']


if world_rank==0:
    print("max_epochs",max_epochs," ",checkpoint_path," ",pretrain_path," ",low_res_dir," ",high_res_dir,"preset",preset," ",out_variable," ",out_var_dict,"lr",lr,"beta_1",beta_1,"beta_2",beta_2,"weight_decay",weight_decay,"warmup_epochs",warmup_epochs,"warmup_start_lr",warmup_start_lr,"eta_min",eta_min,"superres_mag",superres_mag,"cnn_ratio",cnn_ratio,"patch_size",patch_size,"embed_dim",embed_dim,"depth",depth,"decoder_depth",decoder_depth,"num_heads",num_heads,"mlp_ratio",mlp_ratio,"drop_path",drop_path,"drop_rate",drop_rate,"batch_size",batch_size,"num_workers",num_workers,"buffer_size",buffer_size,flush=True)


model_kwargs = {'superres_mag':superres_mag,'cnn_ratio':cnn_ratio,'patch_size':patch_size,'embed_dim':embed_dim,'depth':depth,'decoder_depth':decoder_depth,'num_heads':num_heads,'mlp_ratio':mlp_ratio,'drop_path':drop_path,'drop_rate':drop_rate}


if world_rank==0:
    print("model_kwargs",model_kwargs,flush=True)


if preset!="vit" and preset!="res_slimvit":
    print("Only supports vit or residual slim vit training.",flush=True)
    sys.exit("Not vit or res_slimvit architecture")



# Set up data

variables = dict_in_variables["PRISM"]
 

in_vars = []
for var in variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + "_" + str(level))
    else:
        in_vars.append(var)

#load data module
data_module = cl.data.IterDataModule(
    "downscaling",
    low_res_dir, 
    high_res_dir,
    in_vars,
    out_vars=[out_var_dict[k] for k in out_variable],
    subsample=1,
    batch_size=batch_size,
    buffer_size=buffer_size,
    num_workers=num_workers,
).to(device)

data_module.setup()


temp = [out_var_dict[k] for k in out_variable]

print("temp is ",temp,flush=True)


# Set up deep learning model
model, train_loss,val_losses,test_losses,train_transform,val_transforms,test_transforms = cl.load_downscaling_module(device,data_module=data_module, architecture=preset,model_kwargs=model_kwargs)
  
if dist.get_rank()==0:
    print("train_loss",train_loss,"train_transform",train_transform,flush=True)
 

#denorm = model.test_target_transforms[0]


denorm = test_transforms[0]


print("denorm is ",denorm,flush=True)

checkpoint_file = "/lustre/orion/nro108/scratch/xf9/checkpoints/climate/PRISM_rank_0_epoch_68.ckpt"

if os.path.exists(checkpoint_file):
    print("resume from checkpoint was set to True. Checkpoint path found.",flush=True)

    print("rank",dist.get_rank(),"src_rank",world_rank,flush=True)

    #map_location = 'cuda:'+str(device)
    map_location = 'cpu'

    checkpoint = torch.load(checkpoint_file,map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint

else:
    print("the checkpoint path does not exist.",flush=True)

    sys.exit("checkpoint path does not exist")



model = model.to(device)



if torch.distributed.get_rank()==0:
    print("model is ",model,flush=True)




# Set the model to evaluation mode
model.eval()

cl.utils.visualize.visualize_at_index(
    model,
    data_module,
    out_list=out_variable,
    in_transform=denorm,
    out_transform=denorm,
    variable="tmin",
    src="prism",
    device = device,
    index=0  # visualize the first sample of the test set
)

dist.destroy_process_group()
