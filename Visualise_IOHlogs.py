import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.stats import kendalltau
import seaborn as sbs
import matplotlib.pyplot as plt
import os
from ConfigSpace import ConfigurationSpace

font = {'size'   : 30}

plt.rc('font', **font)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

dim = 5

de_cs = ConfigurationSpace(
    {
        "F": (0.0, 1.0),
        "CR": (0.001, 1.0),
        "lambda_": (10, 10 * dim), # (10, 10 * dims)
        "mutation_base": ["target", "best", "rand"],
        "mutation_reference": ["pbest", "rand", "nan", "best"],
        "mutation_n_comps": [1, 2],
        "use_archive": [False, True],
        "crossover": ["exp", "bin"],
        "lpsr": [False, True],
    }
)

features = [
            *de_cs.keys(),
            "Instance variance",
            "Stochastic variance",
        ]

feature_names_plot = [
    'F',
    'CR',
    'Lambda',
    'Mutation Base',
    'Mutation Reference',
    'Mutation N Comps',
    'Use Archive',
    'Crossover',
    'LPSR',
    "Instance variance",
    "Stochastic variance",
    ]

BASE_DIR = ""
TARGET_DIR = ""
        
os.makedirs(TARGET_DIR, exist_ok=True)

def load_data():
    budgets = [10_000,5_000,2_500,2_000,1_000,500,250,100]
    fids = range(1,25)

    if os.path.exists(f"{BASE_DIR}Thesis_dt_stds.csv"):
        dt_stds = pd.read_csv(f"{BASE_DIR}Thesis_dt_stds.csv", index_col = 0)
    else:
        stds = []
        for budget in tqdm(budgets):
            for fid in fids:
                dt2 = pd.read_csv(f"{BASE_DIR}/F{fid}_B{budget}.csv", index_col=0)
                stds.append([fid, budget, dt2['auc'].std()])
        dt_stds = pd.DataFrame.from_records(stds, columns = ['Fid', 'Budget', 'std'])
        dt_stds['std'] = dt_stds['std'].replace(0, 1)
        dt_stds.to_csv(f"{BASE_DIR}Thesis_dt_stds.csv")

    if os.path.exists(f"{BASE_DIR}Thesis_dt_molt.csv"):
        dt_molt = pd.read_csv(f"{BASE_DIR}Thesis_dt_molt.csv", index_col = 0)
        dt_molt_norm = pd.read_csv(f"{BASE_DIR}Thesis_dt_molt_norm.csv", index_col = 0)
    else:
        records = []
        for budget in tqdm(budgets):
            for fid in fids:
                try:
                    shaps = np.loadtxt(f"{BASE_DIR}/shaps/F{fid}_B{budget}.txt")
                    records.append([fid,budget,*np.mean(np.abs(shaps),axis=0).tolist()])
                except:
                    # If SHAP file is missing, fill with zeros for all features
                    records.append([fid, budget, *([0] * len(features))])
                    print(f"Missing SHAP file for F{fid} B{budget}, filling with zeros.")
        dt_plot = pd.DataFrame.from_records(records, columns=["Fid", "Budget", *features])
        dt_molt = dt_plot.melt(id_vars=['Fid', 'Budget'])
        dt_molt.to_csv(f"{BASE_DIR}Thesis_dt_molt.csv")
        dt_plot_norm = dt_plot.copy()
        dt_plot_norm[dt_plot_norm.columns[2:]] = dt_plot[dt_plot.columns[2:]].div(dt_stds['std'], axis=0)
        dt_molt_norm = dt_plot_norm.melt(id_vars=['Fid', 'Budget'])
        dt_molt_norm.to_csv(f"{BASE_DIR}Thesis_dt_molt_norm.csv")
    dt_molt_norm = dt_molt_norm.replace({x:y for x,y in zip(features, feature_names_plot)})
    dt_molt = dt_molt.replace({x:y for x,y in zip(features, feature_names_plot)})

    return dt_molt, dt_molt_norm

print('Loading data...')
dt_molt, dt_molt_norm = load_data()

def create_shap_over_time(fid, normalize = True, include_legend = True):
    if normalize:
        dt_plotting = dt_molt_norm
    else:
        dt_plotting = dt_molt
    if fid:
        dt_plotting = dt_plotting.query(f"Fid == {fid}")
    if include_legend:
        plt.figure(figsize=(19,9))
        sbs.lineplot(data=dt_plotting.sort_values(['Budget', 'value'], ascending=[False, False]), x='Budget', y='value', hue='variable', style='variable', lw=4, dashes=[(1,0),(1,1),(2,2),(3,1)], markers=True, ms=15, style_order=feature_names_plot, hue_order=feature_names_plot)
        plt.legend(fontsize=20, title='Parameter', loc='lower right', bbox_to_anchor=(1.45, 0))
    else:
        plt.figure(figsize=(16,9))
        sbs.lineplot(data=dt_plotting.sort_values(['Budget', 'variable'], ascending=[False, False]), x='Budget', y='value', hue='variable', style='variable', lw=4, dashes=[(1,0),(1,1),(2,2),(3,1)], markers=True, ms=15, style_order=feature_names_plot, hue_order=feature_names_plot, legend=None)

    # plt.xlim(100, 10^5)
    plt.xscale('log')
    plt.xlabel("Budget")
    plt.ylabel("SHAP value")
    plt.tight_layout()
    if include_legend:
        if fid:
            plt.savefig(f"{TARGET_DIR}/Overtime_SHAP_F{fid}_N{normalize}.png")
        else:
            plt.savefig(f"{TARGET_DIR}/Overtime_SHAP_All_N{normalize}.png")
    else:
        if fid:
            plt.savefig(f"{TARGET_DIR}/Overtime_SHAP_F{fid}_N{normalize}_nolegend.png")
        else:
            plt.savefig(f"{TARGET_DIR}/Overtime_SHAP_All_N{normalize}_nolegend.png")
    plt.close()
print('Creating SHAP over time plots...')
    
for fid in range(1,25):
    for normalize in [True, False]:
        create_shap_over_time(fid, normalize, False)
        create_shap_over_time(fid, normalize, True)
create_shap_over_time(None, True, False)
create_shap_over_time(None, False, False)