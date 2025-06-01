"""Analyse the Modular DE results from the pickle file using fanova in ioh-xplain
line 123 and 124 need to commented out in fanova/visualizer.py
to run for an config with categorical hyperparameters
"""

from config import de_explainer

data_file = r"C:\Users\niels\OneDrive\Documents\Leiden\Year 3\Thesis\dim_5\de_final_5_processed.pkl"
de_explainer.load_results(data_file)

# auc Large
de_explainer.df.loc[de_explainer.df["dim"] == 30, "auc"] = de_explainer.df.loc[
    de_explainer.df["dim"] == 30, "aucLarge"
]

de_explainer.explain_fanova(
    partial_dependence=True,
    best_config=True,
    file_prefix="../output/de_img_fanova/"
)
