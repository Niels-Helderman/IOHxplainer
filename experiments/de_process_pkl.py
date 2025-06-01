"""Process the DE pkl file (fixing mu and lambda)"""

import numpy as np
import pandas as pd
import os

data_file = "de_final_5.pkl"
df = pd.read_pickle(data_file)

df = df.drop(columns=["Unnamed: 0"])


# replacing stuff to fix
df["mutation_reference"] = df["mutation_reference"].replace(np.nan, "nan")
df["adaptation_method"] = df["adaptation_method"].replace(np.nan, "nan")
df["lambda_"] = df["lambda_"].replace(np.nan, "nan")
# df["lambda_"] = df["lambda_"].replace("2", 2.0)
# df["lambda_"] = df["lambda_"].replace("10", 10.0)

# df.loc[(df["lambda_"] == 10.0) & (df["dim"] == 30), "lambda_"] = 300
# df.loc[(df["lambda_"] == 10.0) & (df["dim"] == 5), "lambda_"] = 50
# df.loc[(df["lambda_"] == 2.0) & (df["dim"] == 30), "lambda_"] = 60
# df.loc[(df["lambda_"] == 2.0) & (df["dim"] == 5), "lambda_"] = 10
# df.loc[(df["lambda_"] == "nan") & (df["dim"] == 5), "lambda_"] = 8
# df.loc[(df["lambda_"] == "nan") & (df["dim"] == 30), "lambda_"] = 14

# Save the processed file in the same directory as the original file
output_dir = os.path.dirname(data_file)
output_file = os.path.join(output_dir, "de_final_5_processed.pkl")
df.to_pickle(output_file)
print(f"Processed file saved to: {output_file}")
print(df["lambda_"].describe())
