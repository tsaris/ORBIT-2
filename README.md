**ORBIT Downscaling Experiments Proof of Principle**

Modified from Climate Learn.

(1) Install your conda environment

(2) Then do `pip install -e .` to install the package to the environment

(3) Do  `pip install -r requirements.txt`

(4) Go to tutorial folder. Downscaling.py is the downscaling super resolution proof of principle code

**Run Script for Downscaling and Visualization**


To run it, use the job scheduler to run the job launch_downscaling.sh

`python ./downscaling.py --max_epochs 30 /lustre/orion/lrn036/world-shared/ERA5_npz/5.625_deg/ /lustre/orion/lrn036/world-shared/ERA5_npz/1.40625_deg/ vit t2m`

In the above line --max_epochs is the maximum number of epochs. "/lustre/orion/lrn036/world-shared/ERA5_npz/5.625_deg/" is the input coarse resolution data path. "/lustre/orion/lrn036/world-shared/ERA5_npz/1.40625_deg/" is the ground truth high resolution data path. 
"vit" is an AI architecture of choice. Besides "**vit**", you can also use "**resnet**", "**unet**" or "**res_slimvit**" architecture by setting the corresponding flag.
"t2m" is the output variable. To show how to predict multiple variables, we will use an example with t2m, z500, t850 three variables. then replace "t2m" in the above code line with `["t2m", "z500", "t850"]`.
You can also request multiple nodes by setting `#SBATCH --nodes=1`  and set --nodes to be larger than 1.


(5) To visualize the input and output, run launch_visualize.sh afterwards. Use only a single node with a single GPU.

**Available Training Data**
ERA5 5.6 degree "/lustre/orion/lrn036/world-shared/ERA5_npz/5.625_deg/" 
ERA5 1.4 degree "/lustre/orion/lrn036/world-shared/ERA5_npz/1.40625_deg/"
ERA5 0.25 degree "/lustre/orion/world-shared/lrn036/jyc/frontier/ClimaX-v2/data/ERA5-wb2/0.25_deg" 
