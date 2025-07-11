from miv.data.demand_design import generate_test_demand_design, generate_train_demand_design, generate_z_test_demand_design
import numpy as np
import pandas as pd
from pathlib import Path
import os

train_size= 1000

train_data = generate_train_demand_design(
        data_size=train_size,
        rho=0.5,
        merror_func_str='multi_gaussian',
        m_scale=2.0, 
        n_scale=2.0, 
        bias=0.,
        rand_seed=42
    )

test_data = generate_test_demand_design()


train_data_dict = {}
test_data_dict = {}

for (dat, dat_dict) in ((train_data, train_data_dict), (test_data, test_data_dict)):
    for key in dat.keys():
        if dat.get(key) is not None:
            for i in range(dat[key].shape[1]):
                dat_dict[key + str(i)] = dat[key][:, i]

train_data_df = pd.DataFrame.from_dict(train_data_dict)
test_data_df = pd.DataFrame.from_dict(test_data_dict)

demand_path = Path(__file__).resolve().parent / 'demand' 
os.makedirs(demand_path, exist_ok=True)

train_data_df.to_csv(demand_path / 'train.csv')
test_data_df.to_csv(demand_path / 'test.csv')














