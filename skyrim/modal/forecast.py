import os
import secrets
import subprocess
from modal import App, Image, gpu, Volume, forward
from dotenv import load_dotenv
from datetime import datetime, timedelta
from loguru import logger

load_dotenv()
CDSAPI_KEY = os.getenv("CDSAPI_KEY", "")
CDSAPI_URL = os.getenv("CDSAPI_URL", "")
MODAL_ENV = os.getenv("MODAL_ENV", "prod")
APP_NAME = f"skyrim-forecast-{MODAL_ENV}"
VOLUME_PATH = "/skyrim/outputs"

if not CDSAPI_KEY or not CDSAPI_URL:
    logger.warning("CDS initial conditions disabled, environment not set.")

yesterday = (datetime.now() - timedelta(days=1)).date().isoformat().replace("-", "")

image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3")
    .run_commands(
        "git clone https://github.com/MateusuMelo/skyrim",
        force_build=(MODAL_ENV != "prod"),
    )
    .workdir("/skyrim")
    .run_commands("pip install .")
    .run_commands("pip install -r requirements.txt")
    .run_commands(
        "pip install protobuf==3.20.*",
        "pip install jax==0.4.23 jaxlib==0.4.23 git+https://github.com/deepmind/dm-haiku.git@v0.0.11",
    )
    .run_commands("pip install ngcsdk==3.55.0")
    .run_commands("pip install tblib")

    # Update and install prerequisites
    .run_commands("apt-get update && apt-get install -y --no-install-recommends wget gnupg")

    # Download cuDNN package
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb -O cudnn.deb")

    # Install cuDNN repository
    .run_commands("dpkg -i cudnn.deb")

    # Add GPG key
    .run_commands("cp /var/cudnn-local-repo-ubuntu2204-9.0.0/cudnn-local-960825AE-keyring.gpg /usr/share/keyrings/")

    # Update package lists
    .run_commands("apt-get update")

    # Install cuDNN packages
    .run_commands("apt-get install -y --no-install-recommends libcudnn9-cuda-12 libcudnn9-dev-cuda-12")

    # Clean up
    .run_commands("rm cudnn.deb")
    .env({"LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/nvidia/lib64"})
    .run_commands(
        "pip uninstall -y onnxruntime",
        "pip install onnxruntime-gpu"
    )
    .pip_install("imageio")
    .env(
        {
            "CDSAPI_KEY": CDSAPI_KEY,
            "CDSAPI_URL": CDSAPI_URL,
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_DEFAULT_REGION": "eu-west-1",
        }
    )
)
app = App(APP_NAME)
vol = Volume.from_name("forecasts", create_if_missing=True)


@app.function(
    gpu="A100-40GB",
    scaledown_window=240 * 2,
    timeout=60 * 30,
    image=image,
    volumes={VOLUME_PATH: vol},
)
def run_inference(*args, **kwargs):
    from skyrim.forecast import run_forecast

    output_paths = run_forecast(*args, **kwargs)
    if not kwargs.get("output_dir", "").startswith("s3://"):
        vol.commit()
    logger.success(f"Saved forecasts to {output_paths}!")


analysis_image = (
    Image.debian_slim()
    .pip_install("python-dotenv", "jupyterlab", "loguru", "scipy", "xarray", "zarr", "matplotlib")
    .env(
        {
            "CDSAPI_KEY": CDSAPI_KEY,
            "CDSAPI_URL": CDSAPI_URL,
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_DEFAULT_REGION": "eu-west-1",
        }
    )
)


@app.function(
    volumes={VOLUME_PATH: vol},
    image=image,
    timeout=60 * 60 * 2,  # 2 hour timeout
    memory=(2048, 4096),  # no need more than 4GB
)
def run_analysis():
    vol.reload()
    token = secrets.token_urlsafe(13)
    with forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        print(f"Starting Jupyter at {url}")
        subprocess.run(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                "--port=8888",
                "--LabApp.allow_origin='*'",
                "--LabApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
            stderr=subprocess.DEVNULL,
        )


@app.local_entrypoint()
def main(
        model_name: str = "pangu",
        date: str = yesterday,
        time: str = "0000",
        lead_time: int = 72,
        list_models: bool = False,
        initial_conditions: str = "gfs",
        output_dir: str = VOLUME_PATH,
        filter_vars: str = "",
):
    """
    Run weather forecast inference using specified model and initial conditions.

    Parameters:
        model_name (str): Name of the model for inference. Default is 'pangu'.
        date (str): Date for the forecast start in YYYYMMDD format. Defaults to yesterday.
        time (str): Time for the forecast start in HHMM format. Default is '0000'.
        lead_time (int): Forecast lead time in hours. Default is 6.
        list_models (bool): If True, lists available models. Default is False.
        initial_conditions (str): Source of initial conditions (e.g., 'gfs'). Default is 'gfs'.
        output_dir (str): Directory for saving output. Default is VOLUME_PATH.
        filter_vars (str): Variables to filter in output. Defaults to an empty string.

    Example:
        Run a forecast for 2024-04-20 with a 12-hour lead time using the Pangu model:
        `modal run modal/forecast.py --model pangu --lead-time 12 --date 20240420`

    Note:
        Forecasts typically generate about 2GB of data. Storage costs can be checked at https://modal.com/pricing.
        To analyze the forecast data interactively, execute:
        `modal run modal/forecast.py:run_analysis`
        This command starts a JupyterLab server in a Modal container. For local data analysis, use:
        `modal volume get forecasts /skyrim/outputs/[model_name]/[filename] .[your_local_path]`
        Clean up storage with:
        `modal volume rm forecasts /[model_name] -r`
    """
    # model_name: str = 'pangu', date: str = yesterday, time: str = "0000", lead_time: int = 6, list_models: bool = False, initial_conditions: str = "ifs", output_dir: str = '/skyrim/outputs', filter_vars: str = ''
    run_inference.remote(
        model_name=model_name,
        date=date,
        time=time,
        lead_time=lead_time,
        list_models=list_models,
        initial_conditions=initial_conditions,
        output_dir=output_dir,
        filter_vars=filter_vars,
    )
