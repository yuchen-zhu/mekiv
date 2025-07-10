from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from miv.utils.util import dotdict
from miv.data.demand_design import generate_test_demand_design, generate_train_demand_design, generate_z_test_demand_design
from miv.data.sigmoid_design import generate_test_sigmoid_design, generate_train_sigmoid_design, generate_z_test_sigmoid_design
from miv.data.linear_design import generate_test_linear_design, generate_train_linear_design, generate_z_test_linear_design
from miv.data.linear_design_cp import generate_test_linear_cp_design, generate_train_linear_cp_design, generate_z_test_linear_cp_design
# from miv.data.dahl_lochner import generate_test_dahl_lochner, generate_train_dahl_lochner
# TODO: This function is not working right now because of a pytorch error, uncomment the above to show the error.

from miv.data.data_class import TrainDataSet, TestDataSet, ZTestDataSet


def sim_dgp(design):
    U = design.fu(design.N_data)  # e.g. socio-economic status
    Z = design.fz(design.N_data)  # whether they got a scholarship from some goverment studies - completely at random
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
    if data_name  in ['linear', 'sigmoid', 'demand', 'linear_cp']:
        train_data = generate_train_data(rand_seed=rand_seed, **data_config)
        test_data = generate_test_data(**data_config)

    else:
        data_dir = Path(__file__).resolve().parent
        data_path = data_dir / data_name / 'data.csv'
        config_path = data_dir / data_name / 'config.yaml'

        if data_path.exists and config_path.exists:
            config = yaml.safe_load(stream=config_path.read_text())
            all_dat = pd.read_csv(data_path).rename(columns=config['column_names'])

            train = all_dat.loc['train']

            covariate_labels = [covariate.strip() for covariate in config['covariates'].split(',')]
            X_obs_labels = [x.strip() for x in config['X_obs'].split(',')]

            train_data = TrainDataSet(
                X_hidden=None, 
                X_obs=train[X_obs_labels].values if len(X_obs_labels) >= 0 else None,
                covariate=train[covariate_labels].values if len(covariate_labels) >= 0 else None,
                M=train[['M']].values,
                N=train[['N']].values,
                Z=train[['Z']].values,
                Y = train[['Y']].values
            )

            test = all_dat.loc['test']

            X_hidden_labels = [x.strip() for x in config['X_hidden'].split(',')]
            if len(X_hidden_labels) != 1:
                raise ValueError('This implementation only accommodates 1 mismeasured treatment.')
            X_all_labels = X_hidden_labels + X_obs_labels
            test_data = TestDataSet(
                X_all=test[X_all_labels].values,
                covariate=test[covariate_labels].values,
                Y_struct=test[['Y_struct']].values
            )

            return train_data, test_data

        elif data_path.exists and not config_path.exists:
            raise ValueError(f"User needs to add a config file for processing data.")
        elif not data_path.exist and config_path.exists:
            raise ValueError(f"User needs to add the data csv file.")
        else:
            raise ValueError(f"data name {data_name} does not exist, please implement its simulation or add its data to {data_dir}.")

        
    return train_data, test_data


def generate_train_data(data_name: str, rand_seed: int, **args) -> TrainDataSet:

    if args["n_scale"]:
        n_scale = args["n_scale"]
    else:
        n_scale = args["m_scale"]

    if data_name == "demand":
        return generate_train_demand_design(data_size=args["data_size"], rho=args["rho"],
                                            merror_func_str=args["merror_func_str"],
                                            m_scale=args["m_scale"], n_scale=n_scale,
                                            bias=args["bias"], rand_seed=rand_seed)
    # if data_name == "demand_image":
    #     raise ValueError(f"data name {data_name} is not implemented")

    # if data_name == "dsprite":
    #     raise ValueError(f"data name {data_name} is not implemented")

    if data_name == "sigmoid":
        return generate_train_sigmoid_design(data_size=args["data_size"], merror_func_str=args["merror_func_str"],
                                             m_scale=args["m_scale"], n_scale=n_scale, bias=args["bias"],
                                             rand_seed=rand_seed)
    if data_name == "linear":
        return generate_train_linear_design(data_size=args["data_size"], merror_func_str=args["merror_func_str"],
                                             m_scale=args["m_scale"], n_scale=n_scale, bias=args["bias"],
                                             rand_seed=rand_seed)
        
    if data_name == "linear_cp":
        return generate_train_linear_cp_design(data_size=args["data_size"], merror_func_str=args["merror_func_str"],
                                             m_scale=args["m_scale"], n_scale=n_scale, bias=args["bias"],
                                             rand_seed=rand_seed)
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
        return generate_z_test_demand_design(rho=args['rho'])
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

