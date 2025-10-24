import pandas as pd
import glob


def get_df(sample_frac: float):
    files = glob.glob("../../Data/dhoogla/cicids2017/versions/3/*.parquet")

    df_list = []
    for file in files:
        sub_df = pd.read_parquet(file)
        sampled_sub_df = sub_df.sample(frac=sample_frac, random_state=42)
        df_list.append(sampled_sub_df)

    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    return final_df
