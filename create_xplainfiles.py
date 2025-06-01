import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import ioh
from itertools import product
from functools import partial
from multiprocessing import Pool, cpu_count

import sys
import argparse
import warnings
import os

import pandas as pd

import numpy as np

from copy import copy

import traceback

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
# from ConfigSpace.util import generate_grid
# from IPython.display import display
# from modcma.c_maes import (
#     ModularCMAES,
#     Parameters,
#     Population,
#     mutation,
#     options,
#     parameters,
#     utils,
# )

from modde import ModularDE
from tqdm import tqdm
import glob

from iohxplainer import explainer

import catboost as cb
import shap

from threadpoolctl import threadpool_limits, threadpool_info
import pprint



BASE_DIR = ""
TARGET_DIR = ""
N_PARALLEL = 20
Dimension = 5

catboost_params = {
                "iterations": 100,
                "depth": 14,
            }
de_cs = ConfigurationSpace(
    {
        "F": (0.0, 1.0),
        "CR": (0.001, 1.0),
        "lambda_": (10, (10 * Dimension)), # (10, 10 * dims)
        "mutation_base": ["target", "best", "rand"],
        "mutation_reference": ["pbest", "rand", "nan", "best"],
        "mutation_n_comps": [1, 2],
        "use_archive": [False, True],
        "crossover": ["exp", "bin"],
        # "adaptation_method": ["nan"], # is a constant
        "lpsr": [False, True],
    }
)

df_pickle = pd.read_pickle("de_final_5_processed.pkl")  
grid = df_pickle[de_cs.keys()].drop_duplicates().reset_index(drop=True)


def get_aucs(idx, fid, dim, grid_item, iid, max_budget):
    folder = f"{BASE_DIR}/mod-de-{idx}-{dim}-{fid}-{iid}"
    # try:
    files = glob.glob(f"{folder}/*/*.dat")
    if len(files) == 0:
        print(folder)
        return pd.DataFrame()
    else: 
        file = files[0]
    # print(file)
    try:
        dt = pd.read_csv(file, sep=' ', decimal=',')
        dt = dt[dt['raw_y'] != 'raw_y'].astype(float)
        dt['run_id'] = np.cumsum(dt['evaluations'] == 1)
        records = []
        for run in np.unique(dt['run_id']):
            dt_temp = dt[dt['run_id'] == run]
            new_row = pd.DataFrame({'evaluations' : max_budget, 'raw_y' : min(dt_temp['raw_y']), 'run_id' : run}, index=[0])
            dt_temp = pd.concat([dt_temp, new_row], ignore_index=True)
            dt_temp = dt_temp.query(f"evaluations <= {max_budget}")
            if Dimension == 5:
                auc = np.sum((2 - np.clip(np.log10(dt_temp['raw_y'][:-1]), -8, 2)) * np.ediff1d(dt_temp['evaluations']))/(max_budget*10)
            elif Dimension == 30:
                auc = np.sum((2 - np.clip(np.log10(dt_temp['raw_y'][:-1]), -8, 8)) * np.ediff1d(dt_temp['evaluations']))/(max_budget*10) 
            records.append([fid, iid, dim, run, *grid_item.tolist(), auc])
        df = pd.DataFrame.from_records(records,
                                columns=[
                                    "fid",
                                    "iid",
                                    "dim",
                                    "seed",
                                    *de_cs.keys(),
                                    "auc",
                                ],
                            )
        return df
    except:
        print(idx)
        return pd.DataFrame()

def runParallelFunction(runFunction, arguments):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this system

        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :return:
    """
    

    arguments = list(arguments)
    p = Pool(min(N_PARALLEL, len(arguments)))
    print("Starting parallel processing")
    results = list(tqdm(p.imap(runFunction, arguments), total=len(arguments)))
    p.close()
    return results


def create_intermediate(setting):
    fid, budget = setting
    dim = Dimension
    df_iids = []
    for iid in range(1,6):
        dfs = [get_aucs(idx, fid, dim, grid.iloc[idx], iid, budget) for idx in range(len(grid))]
        df_iids.append(pd.concat(dfs))
    df = pd.concat(df_iids)
    print('saving intermediate')
    df.to_csv(f"{TARGET_DIR}/F{fid}_B{budget}.csv")


def get_shaps(setting):
    fid, budget = setting
    dim = Dimension
    df = pd.read_csv(f"{TARGET_DIR}/F{fid}_B{budget}.csv", index_col=0, keep_default_na=False)
    
    df = df.rename(
        columns={"iid": "Instance variance", "seed": "Stochastic variance"}
    )
    df_display = df.copy(True)

    categorical_columns = df.dtypes[
        (df.dtypes == "object") | (df.dtypes == "category")
    ].index.to_list()
    df[categorical_columns] = df[categorical_columns].apply(
        lambda col: pd.Categorical(col).codes
    )
    df_display[categorical_columns] = df_display[categorical_columns].astype(
        "category"
    )
    
    # for c in categorical_columns:
    # df[c] = df[c].astype('str')
    # df[c] = df[c].astype("category")

    # categorical_columns = df.dtypes[df.dtypes == "category"].index.to_list()



    X = df[
        [
            *de_cs.keys(),
            "Instance variance",
            "Stochastic variance",
        ]
    ]

    y = df["auc"].values
    
    # Skip if all targets are equal
    if np.all(y == y[0]):
        print(f"Skipping F{fid}_B{budget}: all targets are equal ({y[0]})")
        return
    
    bst = cb.CatBoostRegressor(**catboost_params, thread_count=1)
    bst.fit(X, y, cat_features=categorical_columns, verbose=False)
    print("fitted model R2 train:", bst.score(X, y))
    
    explainer = shap.TreeExplainer(bst)
    with threadpool_limits(limits=1):

        shap_values = explainer.shap_values(X)
        # pprint.pprint(threadpool_info())

    # import pickle
    # with open(f"{TARGET_DIR}pkls/F{fid}_B{budget}.pkl", 'wb') as file:
    #     pickle.dump(explainer, file)
    np.savetxt(f"{TARGET_DIR}shaps/F{fid}_B{budget}.txt", shap_values)

def get_shaps_subset(setting, fid, budget):
    nr_samples, seed = setting
    rng = np.random.default_rng(seed)

    df = pd.read_csv(f"{TARGET_DIR}/F{fid}_B{budget}.csv", index_col=0)

    n_configs = int(len(df)/5) #we don't subsample seeds
    sample_idxs = rng.choice(n_configs, nr_samples)
    real_idxs = np.vstack([sample_idxs*5, 
                           (sample_idxs*5)+1, 
                           (sample_idxs*5)+2, 
                           (sample_idxs*5)+3, 
                           (sample_idxs*5)+4]).flatten()
    df = df.iloc[real_idxs]

    df = df.rename(
        columns={"iid": "Instance variance", "seed": "Stochastic variance"}
    )
    df_display = df.copy(True)
    categorical_columns = df.dtypes[
        (df.dtypes == "object") | (df.dtypes == "category")
    ].index.to_list()
    df[categorical_columns] = df[categorical_columns].apply(
        lambda col: pd.Categorical(col).codes
    )
    df_display[categorical_columns] = df_display[categorical_columns].astype(
        "category"
    )
    # for c in categorical_columns:
    # df[c] = df[c].astype('str')
    # df[c] = df[c].astype("category")

    categorical_columns = df.dtypes[df.dtypes == "category"].index.to_list()


    X = df[
        [
            *de_cs.keys(),
            "Instance variance",
            "Stochastic variance",
        ]
    ]

    y = df["auc"].values
    bst = cb.CatBoostRegressor(**catboost_params)
    bst.fit(X, y, cat_features=categorical_columns, verbose=False)
    print("fitted model R2 train:", bst.score(X, y))
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X)

    # import pickle
    # with open(f"{TARGET_DIR}pkls/F{fid}_B{budget}.pkl", 'wb') as file:
    #     pickle.dump(explainer, file)
    np.savetxt(f"{TARGET_DIR}shaps_sub/F{fid}_B{budget}_{nr_samples}_{seed}.txt", shap_values)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)

    if BASE_DIR == "" or TARGET_DIR == "":
        print("Please set the BASE_DIR and TARGET_DIR variables to a valid path")
        sys.exit(1)
        
    os.makedirs(TARGET_DIR, exist_ok=True)
    shap_dir = os.path.join(TARGET_DIR, "shaps")
    os.makedirs(shap_dir, exist_ok=True)
    fids = range(1,25)
    budgets = [10_000,5_000,2_500,2_000,1_000,500,250,100]
    args = product(fids, budgets)
    runParallelFunction(create_intermediate, args)
    # runParallelFunction(get_shaps, args)
    # for fid in range(1,25):
    #     run_func = partial(get_shaps_subset, fid = fid, budget = 10_000)
    #     seeds = [1,2,3]
    #     nr_samples = [500, 1_000, 5_000,10_000,50_000, 100_000]
    #     args = product(nr_samples, seeds)
    #     runParallelFunction(run_func, args)

