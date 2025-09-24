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
import os
import torch
import jax

print("=== DEBUG: Verificando ambiente CUDA ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Nenhuma GPU CUDA disponível")

# Configure environment for GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_PLATFORM_NAME"] = "cuda"

print(f"JAX platform: {jax.lib.xla_bridge.get_backend().platform}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Não definido')}")

# MONKEY PATCH: Corrigir a função torch_to_jax do earth2mip
import earth2mip.networks.graphcast as graphcast_module


def fixed_torch_to_jax(x):
    """Versão corrigida de torch_to_jax que preserva o layout original"""
    print(f"=== DEBUG: Tensor shape original: {x.shape}")

    # NÃO reordenar as dimensões - manter o layout original do GraphCast
    # O modelo espera: [batch, time, channels, lat, lon]
    # O JAX pode lidar com este layout diretamente

    # Converter para numpy primeiro, depois para JAX
    numpy_array = x.cpu().numpy()

    # Manter o layout original: [batch, time, channels, lat, lon]
    # NÃO fazer transpose
    jax_array = jnp.array(numpy_array)
    print(f"=== DEBUG: Conversão via numpy bem-sucedida, shape: {jax_array.shape}")
    return jax_array


# Aplicar o monkey patch
graphcast_module.torch_to_jax = fixed_torch_to_jax
print("=== Monkey patch aplicado para torch_to_jax ===")

# fmt: off
CHANNELS = ["z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700",
            "z850", "z925", "z1000", "q50", "q100", "q150", "q200", "q250", "q300", "q400",
            "q500", "q600", "q700", "q850", "q925", "q1000", "t50", "t100", "t150", "t200",
            "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000", "u50",
            "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850",
            "u925", "u1000", "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500",
            "v600", "v700", "v850", "v925", "v1000", "w50", "w100", "w150", "w200", "w250",
            "w300", "w400", "w500", "w600", "w700", "w850", "w925", "w1000", "u10m", "v10m",
            "t2m", "msl", "tp06",
            ]
# fmt: on

CHANNEL_MAP = [
    ("specific_humidity", "q"),
    ("geopotential", "z"),
    ("temperature", "t"),
    ("u_component_of_wind", "u"),
    ("v_component_of_wind", "v"),
    ("vertical_velocity", "w"),
    ("2m_temperature", "t2m"),
    ("10m_u_component_of_wind", "u10m"),
    ("10m_v_component_of_wind", "v10m"),
    ("mean_sea_level_pressure", "msl"),
    ("toa_incident_solar_radiation", "tp06"),
]


class GraphcastModel(GlobalModel):
    model_name = "graphcast"

    def __init__(self, *args, **kwargs):
        super().__init__(self.model_name, *args, **kwargs)

    def build_model(self):
        model = graphcast.load_time_loop_operational(
            registry.get_model("e2mip://graphcast")
        )
        print(f"=== DEBUG: Model input channels: {model.in_channel_names}")
        print(f"=== DEBUG: Number of input channels: {len(model.in_channel_names)}")
        print(f"=== DEBUG: Model output channels: {model.out_channel_names}")
        print(f"=== DEBUG: Number of output channels: {len(model.out_channel_names)}")
        return model

    @property
    def time_step(self):
        return self.model.time_step

    @property
    def in_channel_names(self):
        return self.model.in_channel_names

    @property
    def out_channel_names(self):
        return self.model.out_channel_names

    def _to_global_da(self, ds: xr.Dataset) -> xr.DataArray:
        """Convert graphcast dataset to our global dataarray (da) format consistent with other models."""
        lvar_map, sfc_map = CHANNEL_MAP[:6], CHANNEL_MAP[6:]
        lvar_dss, sfc_dss = [], []
        ds = ds.squeeze(dim="batch")
        for name, code in lvar_map:
            channels = [f"{code}{l}" for l in list(ds[name].level.values)]
            x = ds[name]
            x["level"] = channels
            x = x.rename({"level": "channel"})
            lvar_dss.append(x)
        for name, code in sfc_map:
            x = ds[name]
            x["channel"] = code
            x = x.expand_dims("channel")
            sfc_dss.append(x)

        global_da = (
            xr.concat(lvar_dss + sfc_dss, dim="channel")
            .transpose("time", "channel", "lat", "lon")
            .as_numpy()
        )
        global_da.name = None
        return global_da

    def _predict_one_step(
            self,
            start_time: datetime.datetime,
            initial_condition: tuple | None = None,
    ) -> xr.DataArray:
        self.stepper = self.model.stepper

        if initial_condition is None:
            print(f"=== DEBUG: Obtendo condições iniciais para {start_time}")
            initial_condition = get_initial_condition_for_model(
                time_loop=self.model,
                data_source=self.data_source,
                time=start_time,
            )

            print(f"=== DEBUG: Tipo da condição inicial: {type(initial_condition)}")
            print(f"=== DEBUG: Shape da condição inicial: {initial_condition.shape}")
            print(f"=== DEBUG: Número de canais: {initial_condition.shape[2]}")

            # Verificar se os canais correspondem ao que o modelo espera
            expected_channels = len(self.model.in_channel_names)
            actual_channels = initial_condition.shape[2]
            print(f"=== DEBUG: Canais esperados: {expected_channels}, Canais obtidos: {actual_channels}")

            if expected_channels != actual_channels:
                print(f"=== DEBUG: AVISO! Número de canais incompatível!")
                print(f"=== DEBUG: Canais esperados: {self.model.in_channel_names}")

            print("=== DEBUG: Inicializando stepper...")
            state = self.stepper.initialize(initial_condition, start_time)
            logger.debug(f"IC fetched - state[0]: {state[0]}")
        else:
            state = initial_condition

        print("=== DEBUG: Executando step...")
        state, output = self.stepper.step(state)
        logger.debug(f"state[0]: {state[0]}")
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