from omegaconf import DictConfig
import wandb
from dotenv import load_dotenv
import os
import flatten_dict
from typing import Optional


_SRC_DIR = os.path.normpath(os.path.dirname(__file__))
WANDB_DIR = os.path.join(_SRC_DIR, "wandb")


def init(
    config: DictConfig, dir: Optional[str] = None, **kwargs
) -> wandb.wandb_sdk.wandb_run.Run:
    """Initialize wandb run.
    
    Args: 
        config: configuration to be logged.
        dir: optional directory to save the wandb run. If not provided, it will be saved in the default directory for wandb.
        kwargs: additional arguments for wandb.init.

    Returns: 
        run: wandb run object.
    """
    # load from the file `.env`
    load_dotenv(dotenv_path=os.path.join(_SRC_DIR, ".env"))

    assert (
        "WANDB_API_KEY" in os.environ
    ), "Please set the WANDB_API_KEY in the .env file."

    assert "project" in kwargs, "Please set a project name."

    config = flatten_dict.flatten(config, reducer="path")

    dir = dir or WANDB_DIR
    os.makedirs(dir, exist_ok=True)

    run = wandb.init(config=config, dir=WANDB_DIR, **kwargs)
    return run


def save_artifact(
    run: wandb.wandb_sdk.wandb_run.Run, local_path: str, name: str, type: str
):
    artifact = wandb.Artifact(name=name, type=type)
    artifact.add_file(local_path=local_path)
    run.log_artifact(artifact)
