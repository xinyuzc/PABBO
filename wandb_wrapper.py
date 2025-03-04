from omegaconf import DictConfig
import wandb
from dotenv import load_dotenv
import os
import flatten_dict


_SRC_DIR = os.path.normpath(os.path.dirname(__file__))
WANDB_DIR = os.path.join(_SRC_DIR, "wandb")


def init(config: DictConfig, **kwargs) -> wandb.wandb_sdk.wandb_run.Run:
    # load from the file `.env`
    load_dotenv(dotenv_path=os.path.join(_SRC_DIR, ".env"))

    assert (
        "WANDB_API_KEY" in os.environ
    ), "Please add `WANDB_API_KEY` to the file `.env`."

    assert "project" in kwargs, "Please set a project name."

    config = flatten_dict.flatten(config, reducer="path")

    os.makedirs(WANDB_DIR, exist_ok=True)

    run = wandb.init(config=config, dir=WANDB_DIR, **kwargs)
    return run


def save_artifact(
    run: wandb.wandb_sdk.wandb_run.Run, local_path: str, name: str, type: str
):
    artifact = wandb.Artifact(name=name, type=type)
    artifact.add_file(local_path=local_path)
    run.log_artifact(artifact)
