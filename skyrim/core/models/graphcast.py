from typing import List
import datetime
import xarray as xr
import jax.numpy as jnp
import numpy as np
import earth2mip.networks.graphcast as graphcast
from pathlib import Path
from earth2mip import registry
from earth2mip.initial_conditions import cds, get_initial_condition_for_model
from ...common import generate_forecast_id, save_forecast
from loguru import logger
from .base import GlobalModel
import torch
import jax
import sys


# ===== RADICAL FIX FOR JAX TENSOR LAYOUT ISSUE =====
def robust_torch_to_jax(tensor):
    """
    Convert PyTorch tensor to JAX array using numpy as intermediate.
    This avoids the dlpack layout compatibility issues.
    """
    # Ensure tensor is on CPU and contiguous
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.contiguous()

    # Convert to numpy first (handles layout issues)
    numpy_array = tensor.numpy()

    # Convert numpy to JAX
    jax_array = jnp.array(numpy_array)

    return jax_array


# Aggressive monkey patching
import earth2mip.networks.graphcast as graphcast_module

graphcast_module.torch_to_jax = robust_torch_to_jax

# Patch all loaded modules
for module_name in list(sys.modules.keys()):
    if hasattr(sys.modules[module_name], 'torch_to_jax'):
        setattr(sys.modules[module_name], 'torch_to_jax', robust_torch_to_jax)

# Also patch the stepper's initialize method if needed
original_initialize = None
if hasattr(graphcast_module, 'GraphCastStepper'):
    original_initialize = graphcast_module.GraphCastStepper.initialize


    def patched_initialize(self, initial_condition, time):
        # Convert tensors to numpy first to avoid layout issues
        if hasattr(initial_condition, 'values'):
            initial_condition.values = initial_condition.values.cpu().contiguous()
        return original_initialize(self, initial_condition, time)


    graphcast_module.GraphCastStepper.initialize = patched_initialize


# ===== END FIX =====

# ... (o resto do seu código CHANNELS e CHANNEL_MAP permanece igual)

class GraphcastModel(GlobalModel):
    model_name = "graphcast"

    def __init__(self, *args, **kwargs):
        super().__init__(self.model_name, *args, **kwargs)
        self._stepper = None

    def build_model(self):
        logger.debug("Loading GraphCast model...")
        model = graphcast.load_time_loop_operational(
            registry.get_model("e2mip://graphcast")
        )
        logger.success("GraphCast model loaded")
        return model

    @property
    @property
    def device(self):
        # GraphCast usa JAX, então o device é controlado pelo JAX
        # Pode retornar string "cpu"/"gpu" ou até jax.devices()[0]
        return str(self.model.device_array.device()) if hasattr(self.model, "device_array") else "jax"

    @property
    def time_step(self):
        # GraphCast tem passo fixo de 6h
        import datetime
        return datetime.timedelta(hours=6)

    @property
    def in_channel_names(self):
        return self.model.in_channel_names

    @property
    def out_channel_names(self):
        return self.model.out_channel_names

    @property
    def stepper(self):
        if self._stepper is None:
            logger.debug("Initializing stepper...")
            self._stepper = self.model.stepper
        return self._stepper

    def _prepare_for_jax(self, data):
        """Prepare data for JAX compatibility"""
        if hasattr(data, 'values') and isinstance(data.values, torch.Tensor):
            # Convert to numpy first, then to JAX compatible format
            data.values = data.values.cpu().contiguous().numpy()
        return data

    def _predict_one_step(
            self,
            start_time: datetime.datetime,
            initial_condition: tuple | None = None,
    ) -> xr.DataArray:

        logger.debug(f"Starting prediction for {start_time}")

        if initial_condition is None:
            logger.debug("Fetching initial conditions...")
            initial_condition = get_initial_condition_for_model(
                time_loop=self.model,
                data_source=self.data_source,
                time=start_time,
            )
            # Prepare for JAX
            initial_condition = self._prepare_for_jax(initial_condition)

        logger.debug("Initializing stepper...")
        try:
            state = self.stepper.initialize(initial_condition, start_time)
            logger.debug("Stepper initialized")

            logger.debug("Taking step...")
            state, output = self.stepper.step(state)
            logger.debug("Step completed")

            return state

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback: try CPU-only execution
            logger.debug("Trying CPU fallback...")
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            state = self.stepper.initialize(initial_condition, start_time)
            state, output = self.stepper.step(state)
            return state

    def forecast(
            self,
            start_time: datetime.datetime,
            n_steps: int = 4,
            channels: List[str] = [],
    ) -> xr.DataArray:
        times = [start_time + i * self.time_step for i in range(n_steps + 1)]
        state, das = None, []
        for n in range(n_steps):
            state = self._predict_one_step(start_time, initial_condition=state)
            logger.success(f"Forecast step {n + 1}/{n_steps} completed")
            da = self._to_global_da(
                state[1] if n == 0 else state[1].isel(time=-1).expand_dims("time")
            )
            if channels:
                da = da.sel(channel=channels)
            da = da.sel(lat=da.lat[::-1])
            das.append(da)
        return xr.concat(das, dim="time").assign_coords(time=times)

    def rollout(
            self,
            start_time: datetime.datetime,
            n_steps: int = 3,
            save: bool = True,
            save_config: dict = {},
    ) -> tuple[xr.DataArray, list[str]]:
        times = [start_time + i * self.time_step for i in range(n_steps + 1)]
        pred, output_paths, source = None, [], self.ic_source
        forecast_id = save_config.get("forecast_id", generate_forecast_id())
        save_config.update({"forecast_id": forecast_id})

        for n in range(n_steps):
            pred = self._predict_one_step(start_time, initial_condition=pred)
            pred_time = start_time + self.time_step
            if save:
                output_path = save_forecast(
                    self._to_global_da(pred[1]).assign_coords(
                        time=[start_time, pred_time]
                    ),
                    self.model_name,
                    start_time,
                    pred_time,
                    source,
                    config=save_config,
                )
                output_paths.append(output_path)
            start_time, source = pred_time, "file"
            logger.success(f"Rollout step {n + 1}/{n_steps} completed")
        return self._to_global_da(pred[1]).assign_coords(time=times[-2:]), output_paths