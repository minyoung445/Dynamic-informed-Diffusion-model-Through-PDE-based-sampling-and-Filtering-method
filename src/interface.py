from typing import Any, Dict, Optional

import hydra
import torch
from omegaconf import DictConfig
from typing import Tuple

from src.datamodules.abstract_datamodule import BaseDataModule
from src.experiment_types._base_experiment import BaseExperiment
from src.utilities.utils import (
    get_logger,
    rename_state_dict_keys_and_save,
)
from src.utilities.wandb_api import reload_checkpoint_from_wandb
from src.experiment_types.interpolation import InterpolationExperiment
from src.utilities.utils import get_local_ckpt_path


"""
In this file you can find helper functions to avoid model/data loading and reloading boilerplate code
"""

log = get_logger(__name__)


def get_lightning_module(config: DictConfig, **kwargs) -> BaseExperiment:
    import pprint
    print("="*60)
    print("[DEBUG] config.model (before instantiate):")
    pprint.pprint(dict(config.model))
    print("="*60)
    r"""Get the ML model, a subclass of :class:`~src.experiment_types._base_experiment.BaseExperiment`, as defined by the key value pairs in ``config.model``.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)
        **kwargs: Any additional keyword arguments for the model class (overrides any key in config, if present)

    Returns:
        BaseExperiment:
            The lightning module that you can directly use to train with pytorch-lightning

    Examples:

    .. code-block:: python

        from src.utilities.config_utils import get_config_from_hydra_compose_overrides

        config_mlp = get_config_from_hydra_compose_overrides(overrides=['model=mlp'])
        mlp_model = get_model(config_mlp)

        # Get a prediction for a (B, S, C) shaped input
        random_mlp_input = torch.randn(1, 100, 5)
        random_prediction = mlp_model.predict(random_mlp_input)
    """
    model = hydra.utils.instantiate(
        config.module,
        model_config=config.model,
        datamodule_config=config.datamodule,
        diffusion_config=config.get("diffusion", default_value=None),
        _recursive_=False,
        **kwargs,
    )

    return model


def get_datamodule(config: DictConfig) -> BaseDataModule:
    r"""Get the datamodule, as defined by the key value pairs in ``config.datamodule``. A datamodule defines the data-loading logic as well as data related (hyper-)parameters like the batch size, number of workers, etc.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)

    Returns:
        Base_DataModule:
            A datamodule that you can directly use to train pytorch-lightning models

    Examples:

    .. code-block:: python

        from src.utilities.config_utils import get_config_from_hydra_compose_overrides

        cfg = get_config_from_hydra_compose_overrides(overrides=['datamodule=icosahedron', 'datamodule.order=5'])
        ico_dm = get_datamodule(cfg)
        print(f"Icosahedron datamodule with order {ico_dm.order}")
    """
    data_module = hydra.utils.instantiate(
        config.datamodule,
        _recursive_=False,
        model_config=config.model,
    )
    return data_module


def get_model_and_data(config: DictConfig) -> Tuple[BaseExperiment, BaseDataModule]:
    r"""Get the model and datamodule. This is a convenience function that wraps around :meth:`get_model` and :meth:`get_datamodule`.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)

    Returns:
        (BaseExperiment, Base_DataModule): A tuple of (module, datamodule), that you can directly use to train with pytorch-lightning

    Examples:

    .. code-block:: python

        from src.utilities.config_utils import get_config_from_hydra_compose_overrides

        cfg = get_config_from_hydra_compose_overrides(overrides=['datamodule=icosahedron', 'model=mlp'])
        mlp_model, icosahedron_data = get_model_and_data(cfg)

        # Use the data from datamodule (its ``train_dataloader()``), to train the model for 10 epochs
        trainer = pl.Trainer(max_epochs=10, devices=1)
        trainer.fit(model=model, datamodule=icosahedron_data)

    """
    data_module = get_datamodule(config)
    model = get_lightning_module(config)
    return model, data_module


def reload_model_from_config_and_ckpt(
    config: DictConfig,
    model_path: str,
    device: Optional[torch.device] = None,
    also_datamodule: bool = True,
    also_ckpt: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    r"""Load a model as defined by ``config.model`` and reload its weights from ``model_path``.

    Args:
        config (DictConfig): The config to use to reload the model
        model_path (str): The path to the model checkpoint (its weights)
        device (torch.device): The device to load the model on. Defaults to 'cuda' if available, else 'cpu'.
        also_datamodule (bool): If True, also reload the datamodule from the config. Defaults to True.
        also_ckpt (bool): If True, also returns the checkpoint from ``model_path``. Defaults to False.

    Returns:
        BaseModel: The reloaded model if load_datamodule is ``False``, otherwise a tuple of (reloaded-model, datamodule)

    Examples:

    .. code-block:: python

        # If you used wandb to save the model, you can use the following to reload it
        from src.utilities.wandb_api import load_hydra_config_from_wandb

        run_path = ENTITY/PROJECT/RUN_ID   # wandb run id (you can find it on the wandb URL after runs/, e.g. 1f5ehvll)
        config = load_hydra_config_from_wandb(run_path, override_kwargs=['datamodule.num_workers=4', 'trainer.gpus=-1'])

        model, datamodule = reload_model_from_config_and_ckpt(config, model_path, load_datamodule=True)

        # Test the reloaded model
        trainer = hydra.utils.instantiate(config.trainer, _recursive_=False)
        trainer.test(model=model, datamodule=datamodule)

    """
    model, data_module = get_model_and_data(config) if also_datamodule else (get_lightning_module(config), None)
    # Reload model
    # device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_state = torch.load(model_path, map_location=device)
    # rename weights (sometimes needed for backwards compatibility)
    state_dict = rename_state_dict_keys_and_save(model_state, model_path)
    # Reload weights
    # remove all keys with model.interpolator prefix
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("model.interpolator")}
    model.load_state_dict(state_dict, strict=False)

    to_return = {
        "model": model,
        "datamodule": data_module,
        "epoch": model_state["epoch"],
        "global_step": model_state["global_step"],
        "wandb": model_state.get("wandb", None),
    }
    if also_ckpt:
        to_return["ckpt"] = model_state
    return to_return


# def get_checkpoint_from_path_or_wandb(
#     model_checkpoint: Optional[torch.nn.Module] = None,
#     model_checkpoint_path: Optional[str] = None,
#     wandb_run_id: Optional[str] = None,
#     model_name: Optional[str] = "model",
#     wandb_kwargs: Optional[Dict[str, Any]] = None,
# ) -> torch.nn.Module:
#     if model_checkpoint is not None:
#         assert model_checkpoint_path is None, "must provide either model_checkpoint or model_checkpoint_path"
#         assert wandb_run_id is None, "must provide either model_checkpoint or wandb_run_id"
#         model = model_checkpoint
#     # elif model_checkpoint_path is not None:
#     #     raise NotImplementedError('Todo: implement loading from checkpoint path')
#     #     assert wandb_run_path is None, 'must provide either model_checkpoint or wandb_run_path'
#     #
#     elif wandb_run_id is not None:
#         # assert model_checkpoint_path is None, 'must provide either wandb_run_path or model_checkpoint_path'
#         override_key_value = ["module.verbose=False"]
#         wandb_kwargs = wandb_kwargs or {}
#         model = reload_checkpoint_from_wandb(
#             run_id=wandb_run_id,
#             also_datamodule=False,
#             override_key_value=override_key_value,
#             local_checkpoint_path=model_checkpoint_path,
#             **wandb_kwargs,
#         )["model"]
#     else:
#         raise ValueError("Provide either model_checkpoint, model_checkpoint_path or wandb_run_id")
#     return model


def get_checkpoint_from_path_or_wandb(
    model_checkpoint: Optional[torch.nn.Module] = None,
    model_checkpoint_path: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
    model_name: Optional[str] = "model",
    wandb_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.nn.Module:
    """
    Interpolator 모델을 다음 세 가지 방법 중 하나로 불러옵니다:
      1) 이미 인스턴스화된 model_checkpoint 객체
      2) 로컬 .ckpt 파일 경로(model_checkpoint_path)
      3) WandB run ID(wandb_run_id) — reload_checkpoint_from_wandb 사용

    Returns:
        torch.nn.Module: 복원된 InterpolationExperiment 인스턴스
    """
    # 1) 직접 전달된 모델 객체
    if model_checkpoint is not None:
        assert model_checkpoint_path is None, "must provide either model_checkpoint or model_checkpoint_path"
        assert wandb_run_id is None,      "must provide either model_checkpoint or wandb_run_id"
        model = model_checkpoint

    # 2) 로컬 체크포인트(.ckpt)에서 로드
    elif model_checkpoint_path is not None:
        assert wandb_run_id is None, "must provide either wandb_run_id or model_checkpoint_path"
        # PyTorch Lightning의 클래스 메서드로 전체 실험 인스턴스를 복원
        model = InterpolationExperiment.load_from_checkpoint(model_checkpoint_path)
        model.eval()  # 평가 모드로 전환

    # 3) WandB에서 체크포인트 로드
    elif wandb_run_id is not None:
        override_key_value = ["module.verbose=False"]
        wandb_kwargs = wandb_kwargs or {}
        loaded = reload_checkpoint_from_wandb(
            run_id=wandb_run_id,
            also_datamodule=False,
            override_key_value=override_key_value,
            local_checkpoint_path=model_checkpoint_path,
            **wandb_kwargs,
        )
        model = loaded["model"]
        model.eval()

    else:
        raise ValueError("Provide either model_checkpoint, model_checkpoint_path or wandb_run_id")

    return model