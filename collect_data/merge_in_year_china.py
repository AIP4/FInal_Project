import os
import pandas as pd
from tqdm import tqdm
import glob

years = [2018, 2019, 2020, 2021, 2022]
locations = ["chifeng", "dalian", "qingdao", "tongliao", "yanan"]
target_dir = "filteerd/kma_3H"

for year in years:
    dataframes = []
    for location in locations:
        df = pd.read_csv(f"filtered/cma_1H/{year}/{location}.csv")
        new_column_names = {col: f"{col}_{location}" for col in df.columns if col != "TM"}
        df.rename(columns=new_column_names, inplace=True)
        dataframes.append(df)

    merged_data = dataframes[0]
    for df in dataframes[1:]:
        merged_data = pd.merge(merged_data, df, on="TM", how="inner")

    merged_data.sort_values(by="TM", inplace=True)

    output_file = f"filtered/cma_1H/{year}/{year}.csv"
    merged_data.to_csv(output_file, index=False)
