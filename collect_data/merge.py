import pandas as pd
import os
import re

def get_csv_files(path="./data/kma/"):
    all_files = os.listdir(path)
    csv_files = [file for file in all_files if re.search(r"\.csv$", file)]
    return csv_files

csv_files = get_csv_files()

years = ["2018", "2019", "2020", "2021", "2022"]
for year in years:
    sub_csv_files = [csv_file for csv_file in csv_files if year in csv_file]
    
    df = pd.DataFrame()
    for filepath in sub_csv_files:
        df_tmp = pd.read_csv(filepath, encoding="utf-8-sig")
        df = pd.concat([df, df_tmp], axis=0)
    save_path = "./data/kma/" + f"kma_{year}" + ".csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")