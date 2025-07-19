from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
import ray
import logging
import torch

from miv.utils.util import grid_search_dict, make_dotdict, dotdict
from miv.models.MerrorKIV.trainer import MerrorKIVTrainer
from miv.models.KIV_M.trainer import KIV_MTrainer
from miv.models.KIV_N.trainer import KIV_NTrainer
from miv.models.KIV_MN.trainer import KIV_MNTrainer
from miv.models.KIV_X.trainer import KIV_XTrainer
from miv.models.base_KIV.trainer import BaseKIVTrainer

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
    data_param: Dict[str, Any],
    train_config: Dict[str, Any],
    experiment_id: int
):
    Train_cls = get_trainer(alg_name)
    trainer = Train_cls(
        data_configs=data_param, 
        train_params=train_config
    )
    out = trainer.train(
        rand_seed=experiment_id
        
    )
    return out


def experiments(
    alg_name: str,
    configs: Dict[str, Any],
    dump_dir: Path,
    num_cpus: int,
    num_gpus: Optional[int],
):
    train_config = configs["train_params"]
    org_data_config = configs["data"]
    n_repeat: int = configs["n_repeat"]

    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

    use_gpu = (0 if num_gpus is None else num_gpus) > 0

    if use_gpu and torch.cuda.is_available():
        remote_run = ray.remote(num_gpus=num_gpus, max_calls=1)(run_one)
    else:
        remote_run = ray.remote(run_one)


    for dump_name, data_param in grid_search_dict(org_data_config):
        dump_name = org_data_config["data_name"] + "_" + dump_name

        tasks = [
            remote_run.remote(
                    alg_name=alg_name,
                    data_param=data_param,
                    train_config=train_config,
                    experiment_id=idx
            )
            for idx in range(n_repeat)
        ]

        results = ray.get(tasks)

        one_dump_dir = dump_dir / dump_name
        os.mkdir(one_dump_dir)

        assert all(len(result) == 4 for result in results)

        mse_list = [
            mse for mse, _, _, _ in results
        ]
        test_pred = [
            test_preds.flatten() for _, _, test_preds, _ in results
        ]

        np.savetxt(one_dump_dir / "mse.csv", X=np.array(mse_list))
        np.savetxt(one_dump_dir / "test_pred.csv", X=np.array(test_pred).T)
        np.savetxt(one_dump_dir / "test_input.csv", X=results[0][1])
        np.savetxt(one_dump_dir / "test_label.csv", X=results[0][3])


        logger.critical(f"{dump_name} ended")

    ray.shutdown()
