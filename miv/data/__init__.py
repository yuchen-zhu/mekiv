from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import numpy as np

from miv.utils.util import dotdict
from miv.data.demand_design import (
    generate_test_demand_design,
    generate_train_demand_design,
    generate_z_test_demand_design,
)
from miv.data.sigmoid_design import (
    generate_test_sigmoid_design,
    generate_train_sigmoid_design,
    generate_z_test_sigmoid_design,
)
from miv.data.linear_design import (
    generate_test_linear_design,
    generate_train_linear_design,
    generate_z_test_linear_design,
)
from miv.data.linear_design_cp import (
    generate_test_linear_cp_design,
    generate_train_linear_cp_design,
    generate_z_test_linear_cp_design,
)

# from miv.data.dahl_lochner import generate_test_dahl_lochner, generate_train_dahl_lochner
# TODO: This function is not working right now because of a pytorch error, uncomment the above to show the error.

from miv.data.data_class import TrainDataSet, TestDataSet, ZTestDataSet


def sim_dgp(design):
    U = design.fu(design.N_data)  # e.g. socio-economic status
    Z = design.fz(
        design.N_data
    )  # whether they got a scholarship from some goverment studies - completely at random
    X = design.fx(Z, U, design.N_data)  # IQ/'ability'
    M = design.fm(X, design.N_data)
    N = design.fn(X, design.N_data)
    Y = design.fy(X, U, design.N_data)
    # breakpoint()
    data = dotdict({})
    data.X = X
    data.Y = Y
    data.Z = Z
    data.M = M
    data.N = N
    data.U = U
    return data


def load_data(data_config, rand_seed: int = 42):
    data_name = data_config["data_name"]
    if data_name in ["linear", "sigmoid", "demand", "linear_cp"]:
        train_data = generate_train_data(rand_seed=rand_seed, **data_config)
        test_data = generate_test_data(**data_config)
        return train_data, test_data

    else:
        data_dir = Path(__file__).resolve().parent
        train_path = data_dir / data_name / "train.csv"
        test_path = data_dir / data_name / "test.csv"
        config_path = data_dir / data_name / "config.yaml"

        required_paths = [train_path, test_path, config_path]
        exist_paths = map(Path.exists, required_paths)

        if not all(exist_paths):
            error_message = "Missing required files: "
            error_message += ", ".join(
                p.name for p, exists in zip(required_paths, exist_paths) if not exists
            )
            raise ValueError(error_message)

        config = yaml.safe_load(stream=config_path.read_text())
        train_config = config["train"]
        test_config = config["test"]
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        train_labels = {
            key: [
                label
                for label in map(str.strip, train_config[key].split(","))
                if len(label) > 0
            ]
            for key in train_config.keys()
            if train_config[key] is not None
        }
        if len(train_labels["X_hidden"]) != 1:
            raise ValueError(
                "This implementation only accommodates 1 mismeasured treatment."
            )

        test_labels = {
            key: [
                label
                for label in map(str.strip, test_config[key].split(","))
                if len(label) > 0
            ]
            for key in test_config.keys()
            if test_config[key] is not None
        }

        train_data = TrainDataSet(
            X_hidden=(
                train[train_labels["X_hidden"]].values
                if train_labels.get("X_hidden") is not None
                else None
            ),
            X_obs=(
                train[train_labels["X_obs"]].values
                if train_labels.get("X_obs") is not None
                else None
            ),
            covariate=(
                train[train_labels["covariate"]].values
                if train_labels.get("covariate") is not None
                else None
            ),
            M=(
                train[train_labels["M"]].values
                if train_labels.get("M") is not None
                else None
            ),
            N=(
                train[train_labels["N"]].values
                if train_labels.get("N") is not None
                else None
            ),
            Z=(
                train[train_labels["Z"]].values
                if train_labels.get("Z") is not None
                else None
            ),
            Y=(
                train[train_labels["Y"]].values
                if train_labels.get("Y") is not None
                else None
            ),
            Y_struct=(
                train[train_labels["Y_struct"]].values
                if train_labels.get("Y_struct") is not None
                else None
            ),
        )

        test_data = TestDataSet(
            X_all=(
                test[test_labels["X_all"]].values
                if test_labels.get("X_all") is not None
                else None
            ),
            covariate=(
                test[test_labels["covariate"]].values
                if test_labels.get("covariate") is not None
                else None
            ),
            Y_struct=(
                test[test_labels["Y_struct"]].values
                if test_labels.get("Y_struct") is not None
                else None
            ),
        )

        return train_data, test_data


def generate_train_data(data_name: str, rand_seed: int, **args) -> TrainDataSet:

    if args["n_scale"]:
        n_scale = args["n_scale"]
    else:
        n_scale = args["m_scale"]

    if data_name == "demand":
        return generate_train_demand_design(
            data_size=args["data_size"],
            rho=args["rho"],
            merror_func_str=args["merror_func_str"],
            m_scale=args["m_scale"],
            n_scale=n_scale,
            bias=args["bias"],
            rand_seed=rand_seed,
        )
    # if data_name == "demand_image":
    #     raise ValueError(f"data name {data_name} is not implemented")

    # if data_name == "dsprite":
    #     raise ValueError(f"data name {data_name} is not implemented")

    if data_name == "sigmoid":
        return generate_train_sigmoid_design(
            data_size=args["data_size"],
            merror_func_str=args["merror_func_str"],
            m_scale=args["m_scale"],
            n_scale=n_scale,
            bias=args["bias"],
            rand_seed=rand_seed,
        )
    if data_name == "linear":
        return generate_train_linear_design(
            data_size=args["data_size"],
            merror_func_str=args["merror_func_str"],
            m_scale=args["m_scale"],
            n_scale=n_scale,
            bias=args["bias"],
            rand_seed=rand_seed,
        )

    if data_name == "linear_cp":
        return generate_train_linear_cp_design(
            data_size=args["data_size"],
            merror_func_str=args["merror_func_str"],
            m_scale=args["m_scale"],
            n_scale=n_scale,
            bias=args["bias"],
            rand_seed=rand_seed,
        )
    else:
        raise ValueError(f"data name {data_name} is not implemented")


def generate_test_data(data_name: str, **args) -> TestDataSet:
    if data_name == "demand":
        return generate_test_demand_design()
    elif data_name == "demand_image":
        raise ValueError(f"data name {data_name} is not implemented")
    elif data_name == "dsprite":
        raise ValueError(f"data name {data_name} is not implemented")
    elif data_name == "sigmoid":
        return generate_test_sigmoid_design()
    # elif data_name == "dahl_lochner":
    #     return generate_test_dahl_lochner()
    elif data_name == "linear":
        return generate_test_linear_design()
    elif data_name == "linear_cp":
        return generate_test_linear_cp_design()
    else:
        raise ValueError(f"data name {data_name} is not implemented")


def generate_z_test_data(data_name: str, **args) -> ZTestDataSet:
    if data_name == "demand":
        return generate_z_test_demand_design(rho=args["rho"])
    elif data_name == "demand_image":
        raise ValueError(f"data name {data_name} is not implemented")
    elif data_name == "dsprite":
        raise ValueError(f"data name {data_name} is not implemented")
    elif data_name == "sigmoid":
        return generate_z_test_sigmoid_design()
    elif data_name == "linear":
        return generate_z_test_linear_design()
    elif data_name == "linear_cp":
        return generate_z_test_linear_cp_design()
    elif data_name == "dahl_lochner":
        pass
    else:
        raise ValueError(f"data name {data_name} is not implemented")
    

def split_train_data(split_ratio: float, train_data: TrainDataSet):
    n_data = train_data.X_hidden.shape[0]
    idx_train_1st, idx_train_2nd = train_test_split(
        np.arange(n_data), train_size=split_ratio
    )

    def get_data(data, idx):
        return data[idx] if data is not None else None

    train_1st_data, train_2nd_data = {}, {}
    for key in train_data.keys():
        train_1st_data[key], train_2nd_data[key] = get_data(
            train_data[key], idx_train_1st
        ), get_data(train_data[key], idx_train_2nd)

    train_1st_data, train_2nd_data = TrainDataSet(**train_1st_data), TrainDataSet(
        **train_2nd_data
    )
    return train_1st_data, train_2nd_data
