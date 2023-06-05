from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
import ray
import logging
import torch

from miv.util import grid_search_dict, make_dotdict, dotdict
from miv.models.MerrorKIV.trainer import MerrorKIVTrainer
# from miv.models.LVM.trainer_old import LVMTrainer
# from miv.models.LVM.trainer_cp import LVMTrainer
from miv.models.LVM.trainer import LVMTrainer
from miv.models.KIV_M.trainer import KIV_MTrainer
from miv.models.KIV_N.trainer import KIV_NTrainer
from miv.models.KIV_MN.trainer import KIV_MNTrainer
from miv.models.KIV_X.trainer import KIV_XTrainer
from miv.models.base_KIV.trainer import BaseKIVTrainer

logger = logging.getLogger()


def get_trainer(alg_name: str):
    if alg_name == "LVM":
        return LVMTrainer
    elif alg_name == "MerrorKIV":
        return MerrorKIVTrainer
    elif alg_name == "KIV_M":
        return KIV_MTrainer
    elif alg_name == "KIV_N":
        return KIV_NTrainer
    elif alg_name == "KIV_MN":
        return KIV_MNTrainer
    elif alg_name == "KIV_oracle":
        return KIV_XTrainer
    # elif alg_name == "oracle_KIV":
    #     return KernelIVTrainer
    else:
        raise ValueError(f"invalid algorithm name {alg_name}")


def run_one(alg_name: str, data_param: Dict[str, Any], train_config: Dict[str, Any],
            use_gpu: bool, dump_dir_root: Optional[Path], experiment_id: int, verbose: int):
    Train_cls = get_trainer(alg_name)
    one_dump_dir = None
    if dump_dir_root is not None:
        one_dump_dir = dump_dir_root.joinpath(f"{experiment_id}/")
        os.mkdir(one_dump_dir)
    trainer = Train_cls(data_param, train_config, use_gpu, one_dump_dir)
    out = trainer.train(experiment_id, verbose)
    # breakpoint()
    return out


def experiments(alg_name: str,
                configs: Dict[str, Any],
                dump_dir: Path,
                num_cpus: int, num_gpu: Optional[int]):
    train_config = configs["train_params"]
    org_data_config = configs["data"]
    n_repeat: int = configs["n_repeat"]

    if num_cpus <= 1:
        ray.init(local_mode=True, num_gpus=num_gpu)
        verbose: int = 2
    else:
        ray.init(num_cpus=num_cpus, num_gpus=num_gpu)
        verbose: int = 0

    # use_gpu: bool = (num_gpu is not None)
    use_gpu = False

    # if use_gpu and torch.cuda.is_available():
    #     remote_run = ray.remote(num_gpus=1, max_calls=1)(run_one)
    # else:
    remote_run = ray.remote(run_one)

    for dump_name, data_param in grid_search_dict(org_data_config):
        dump_name = org_data_config["data_name"] + "_" + dump_name
        # breakpoint()
        one_dump_dir = dump_dir.joinpath(dump_name)
        os.mkdir(one_dump_dir)
        tasks = [remote_run.remote(alg_name, data_param, train_config,
                                   use_gpu, one_dump_dir, idx, verbose) for idx in range(n_repeat)]
        # breakpoint()
        res = ray.get(tasks)
        process_res(res, one_dump_dir)
        # breakpoint()
        # np.savetxt(one_dump_dir.joinpath("results.csv"), X=np.array(res))
        # np.savetxt(one_dump_dir.joinpath("mse.csv"), X=mse_array)
        # np.savetxt(one_dump_dir.joinpath("test_input.csv"), X=test_input)
        # np.savetxt(one_dump_dir.joinpath("test_pred.csv"), X=test_pred)
        # np.savetxt(one_dump_dir.joinpath("test_label.csv"), X=test_label)
        logger.critical(f"{dump_name} ended")

    ray.shutdown()


def process_res(res, one_dump_dir):
    """
    process results to a format that we can save
    :param res: list of tuples. Each tuple: (mses, test_inputs, test_preds, test_labels)
            one_dump_dir: the dir to dump the results
    :return: a list of mse, an array of test inputs and an array of test preds
    """
    # breakpoint()
    assert (len(res[0]) == 4)
    if isinstance(res[0][0], dict):
        mse_list, z_mse_list = [], []
        test_input, z_test_input = res[0][1]['x'], res[0][1]['z']
        test_pred, z_test_pred = [], []
        test_label, z_test_label = res[0][3]['x'], res[0][3]['z']
        for tup in res:
            mse_list.append(tup[0]['x'])
            # z_mse_list.append(tup[0]['z'])
            z_mse_list.append(np.array([1.]))
            test_pred.append(tup[2]['x'].flatten())
            # z_test_pred.append(tup[2]['z'].flatten())
            z_test_pred.append(np.array([1.]))

        np.savetxt(one_dump_dir.joinpath("mse.csv"), X=np.array(mse_list))
        # np.savetxt(one_dump_dir.joinpath("z_mse.csv"), X=np.array(z_mse_list))
        np.savetxt(one_dump_dir.joinpath("test_input.csv"), X=test_input)
        # np.savetxt(one_dump_dir.joinpath("z_test_input.csv"), X=z_test_input)
        np.savetxt(one_dump_dir.joinpath("test_pred.csv"), X=np.array(test_pred).T)
        # np.savetxt(one_dump_dir.joinpath("z_test_pred.csv"), X=np.array(z_test_pred).T)
        # breakpoint()
        np.savetxt(one_dump_dir.joinpath("test_label.csv"), X=test_label)
        # np.savetxt(one_dump_dir.joinpath("z_test_label.csv"), X=z_test_label)
    else:
        mse_list = []
        test_input = res[0][1]
        test_pred = []
        test_label = res[0][3]
        for tup in res:
            mse_list.append(tup[0])

            test_pred.append(tup[2].flatten())


        np.savetxt(one_dump_dir.joinpath("mse.csv"), X=np.array(mse_list))

        np.savetxt(one_dump_dir.joinpath("test_input.csv"), X=test_input)

        np.savetxt(one_dump_dir.joinpath("test_pred.csv"), X=np.array(test_pred).T)

        np.savetxt(one_dump_dir.joinpath("test_label.csv"), X=test_label)

