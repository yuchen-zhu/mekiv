import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray
import torch

from miv.models.KIV.trainer import (
    KIV_MTrainer,
    KIV_MNTrainer,
    KIV_NTrainer,
    KIV_XTrainer,
)
from miv.models.MerrorKIV.trainer import MerrorKIVTrainer
from miv.utils import DotDict, grid_search_dict

logger = logging.getLogger(__name__)


def get_trainer(alg_name: str):
    if alg_name == "MerrorKIV":
        return MerrorKIVTrainer
    elif alg_name == "KIV_M":
        return KIV_MTrainer
    elif alg_name == "KIV_N":
        return KIV_NTrainer
    elif alg_name == "KIV_MN":
        return KIV_MNTrainer
    elif alg_name == "KIV_oracle":
        return KIV_XTrainer
    else:
        raise NotImplementedError(f"invalid algorithm name {alg_name}")


def run_one(
    alg_name: str,
    data_param: dict[str, Any],
    train_params: dict[str, Any],
    experiment_id: int,
):
    train_class = get_trainer(alg_name)
    trainer = train_class(
        data_configs=DotDict.from_dict(data_param),
        train_params=DotDict.from_dict(train_params),
    )
    out = trainer.train(rand_seed=experiment_id)
    return out


def experiments(
    alg_name: str,
    configs: dict[str, Any],
    dump_dir: Path,
    num_cpus: int,
    num_gpus: Optional[int],
):
    train_params = configs["train_params"]
    org_data_config = configs["data"]
    n_repeat = configs["n_repeat"]

    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

    use_gpu = num_gpus is not None and num_cpus > 0

    if use_gpu and torch.cuda.is_available():
        remote_run = ray.remote(num_gpus=num_gpus, max_calls=1)(run_one)
    else:
        remote_run = ray.remote(run_one)

    for dump_name, data_param in grid_search_dict(org_data_config):
        dump_name = f"{org_data_config['data_name']}_{dump_name}"

        tasks = [
            remote_run.remote(
                alg_name=alg_name,
                data_param=data_param,
                train_params=train_params,
                experiment_id=idx,
            )
            for idx in range(n_repeat)
        ]

        results = ray.get(tasks)

        one_dump_dir = dump_dir / dump_name
        one_dump_dir.mkdir()

        assert all(len(result) == 4 for result in results)

        mse_list, _, test_preds, _ = zip(*results)
        test_preds = [test_pred.flatten() for test_pred in test_preds]

        np.savetxt(one_dump_dir / "mse.csv", X=np.array(mse_list))
        np.savetxt(one_dump_dir / "test_pred.csv", X=np.array(test_preds).T)
        np.savetxt(one_dump_dir / "test_input.csv", X=results[0][1])
        np.savetxt(one_dump_dir / "test_label.csv", X=results[0][3])

        logger.info(f"{dump_name} ended")

    ray.shutdown()
