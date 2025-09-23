import time
import datetime
from typing import List
from loguru import logger
import xarray as xr
from earth2studio.models.px import FuXi
from earth2studio.data import GFS, IFS, CDS
from earth2studio.io import XarrayBackend
import earth2studio.run as run
from .base import GlobalModel

# fmt: off
CHANNELS = ["z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700",
            "z850", "z925", "z1000", "t50", "t100", "t150", "t200", "t250", "t300", "t400",
            "t500", "t600", "t700", "t850", "t925", "t1000", "u50", "u100", "u150", "u200",
            "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925", "u1000", "v50",
            "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850",
            "v925", "v1000", "r50", "r100", "r150", "r200", "r250", "r300", "r400", "r500",
            "r600", "r700", "r850", "r925", "r1000", "t2m", "u10m", "v10m", "msl", "tp"
            ]


# fmt: on


class FuxiModel(GlobalModel):
    """
    From:
    https://github.com/NVIDIA/earth2studio/blob/68dd00bd76be8abc90badd39d0f51f26294ce526/earth2studio/models/px/fuxi.py#L116-L120

        FuXi weather model consists of three auto-regressive U-net transfomer models with
        a time-step size of 6 hours. The three models are trained to predict short (5days),
        medium (10 days) and longer (15 days) forecasts respectively. FuXi operates on
        0.25 degree lat-lon grid (south-pole including) equirectangular grid with 70
        atmospheric/surface variables. This model uses two time-steps as an input.

    - https://arxiv.org/abs/2306.12873
        The performance evaluation demonstrates that FuXi has forecast performance comparable to ECMWF
        ensemble mean (EM) in 15-day forecasts. FuXi surpasses the skillful forecast lead time achieved 
        by ECMWF HRES by extending the lead time for Z500 from 9.25 to 10.5 days and 
        for T2M from 10 to 14.5 days.

    - https://github.com/tpys/FuXi
        The variable 'Z' represents geopotential and not geopotential height.
        The variable 'TP' represents total precipitation accumulated over a period of 6 hours.

    """

    model_name = "fuxi"

    def __init__(self, *args, **kwargs):
        super().__init__(self.model_name, *args, **kwargs)

    def build_model(self):
        return FuXi.load_model(FuXi.load_default_package())

    def build_datasource(self):
        if self.ic_source == "gfs":
            return GFS()
        elif self.ic_source == "ifs":
            return IFS()
        elif self.ic_source == "cds":
            return CDS()

    @property
    def time_step(self):
        return datetime.timedelta(hours=6)

    @property
    def in_channel_names(self):
        # TODO: add the input channel names
        return CHANNELS

    @property
    def out_channel_names(self):
        # TODO: add the output channel names
        return CHANNELS

    def forecast(
            self,
            start_time: datetime.datetime,
            n_steps: int,
            channels: List[str] = [],
    ) -> xr.DataArray:

        io = XarrayBackend({})
        io = run.deterministic(
            time=[start_time],
            nsteps=n_steps,
            prognostic=self.model,
            data=self.data_source,
            io=io,
        )
        # TODO: this transformation takes forever, need to optimize
        ts = time.time()
        da = io.root.squeeze().to_array()
        logger.debug(f"io to xr.DataArray {time.time() - ts:.1f} seconds")
        # returned da has the following structure (i.e., earth2studio style):
        # >> da.dims
        # ('variable', 'lead_time', 'lat', 'lon')
        # >> da.coords
        # Coordinates:
        #     time       datetime64[ns] 2024-04-01
        #   * lead_time  (lead_time) timedelta64[ns] 00:00:00 06:00:00 ... 1 days 00:00:00
        #   * lat        (lat) float64 90.0 89.75 89.5 89.25 ... -89.25 -89.5 -89.75 -90.0
        #   * lon        (lon) float64 0.0 0.25 0.5 0.75 1.0 ... 359.0 359.2 359.5 359.8
        #   * variable   (variable) object 'z50' 'z100' 'z150' ... 'v10m' 'msl' 'tp'

        # arrange the dataarray in the format of time, channel, lat, lon, i.e. skyrim style
        da = (
            da.rename({"variable": "channel"})
            .assign_coords(time=lambda x: x.time + x.lead_time)
            .swap_dims({"lead_time": "time"})
            .drop_vars("lead_time")
            .transpose("time", "channel", "lat", "lon")
            .astype("float32")
        )

        return da.sel(channel=channels) if channels else da

    def rollout(self, start_time: datetime.datetime, n_steps: int, channels: List[str] = None) -> tuple[
        xr.DataArray, list[str]]:
        """Perform a model rollout (forecast) starting from the given time."""
        if channels is None:
            channels = []

        da = self.forecast(start_time, n_steps, channels)
        return da, channels

    def predict_one_step(self, start_time: datetime.datetime, channels: List[str] = None) -> xr.DataArray:
        """Predict one time step forward."""
        if channels is None:
            channels = []

        return self.forecast(start_time, n_steps=1, channels=channels)
