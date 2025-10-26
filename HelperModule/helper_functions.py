import pandas as pd
import glob
import os


def get_df(sample_frac: float):
    helper_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(helper_dir)
    data_path_pattern = os.path.join(
        project_root, "Data", "dhoogla", "cicids2017", "versions", "3", "*.parquet"
    )
    files = glob.glob(data_path_pattern)

    df_list = []
    for file in files:
        sub_df = pd.read_parquet(file)
        sampled_sub_df = sub_df.sample(frac=sample_frac, random_state=42)
        df_list.append(sampled_sub_df)

    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    label_counts = final_df["Label"].value_counts()
    least_frequent_labels = label_counts.nsmallest(6).index.tolist()

    new_label_col = final_df["Label"].copy()
    new_label_col.loc[final_df["Label"].isin(least_frequent_labels)] = "Other_attacks"
    final_df["Label"] = new_label_col
    return final_df
