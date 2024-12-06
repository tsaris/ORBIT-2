import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)
# assuming we are downscaling geopotential from ERA5

# Set up data
variables = [
    "land_sea_mask",
    "orography",
    "lattitude",
#    "toa_incident_solar_radiation",
    "2m_temperature",
#    "10m_u_component_of_wind",
#    "10m_v_component_of_wind",
    "geopotential",
    "temperature",
#    "relative_humidity",
    "specific_humidity",
#    "u_component_of_wind",
#    "v_component_of_wind",
]
out_var_dict = {
    "t2m": "2m_temperature",
#    "z500": "geopotential_500",
#    "t850": "temperature_850",
}
in_vars = []
for var in variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + "_" + str(level))
    else:
        in_vars.append(var)


dm = cl.data.IterDataModule(
    "downscaling",
    "/lustre/orion/lrn036/world-shared/ERA5_npz/5.625_deg", 
    "/lustre/orion/lrn036/world-shared/ERA5_npz/1.40625_deg",
    in_vars,
    out_vars=[out_var_dict["t2m"]],
    subsample=1,
    batch_size=32,
    buffer_size=1000,
    num_workers=1,
)

dm.setup()

print("dm.hparams",dm.hparams,flush=True)


# Set up deep learning model
model = cl.load_downscaling_module(data_module=dm, architecture="vit")

denorm = model.test_target_transforms[0]



model = cl.LitModule.load_from_checkpoint(
    "/lustre/orion/nro108/proj-shared/xf9/climate-learn/tutorial/vit_downscaling_t2m/checkpoints/epoch_011.ckpt",
    net=model.net,
    optimizer=model.optimizer,
    lr_scheduler=None,
    train_loss=None,
    val_loss=None,
    test_loss=model.test_loss,
    test_target_tranfsorms=model.test_target_transforms,
)



cl.utils.visualize.visualize_at_index(
    model,
    dm,
    in_transform=denorm,
    out_transform=denorm,
    variable="2m_temperature",
    src="era5",
    index=0  # visualize the first sample of the test set
)
