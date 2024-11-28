import pandas as pd
import os
import re

def get_csv_files(path="./data/kma/"):
    all_files = os.listdir(path)
    csv_files = [file for file in all_files if re.search(r"\.csv$", file)]
    return csv_files

csv_files = get_csv_files()

years = ["2018", "2019", "2020", "2021", "2022"]
citys = ["andong", "daegu", "gwangju", "jeonju", "seoul"]
for year in years:
    for city in citys:
        sub_csv_files = [f"./data/kma/{year}/finedust/kma_{year}_{city}.csv", 
                         f"./data/kma/{year}/weather/kma_{year}_{city}.csv"]
        
        df = pd.DataFrame()
        print(f"concating: {sub_csv_files}")
        for filepath in sub_csv_files:
            df_tmp = pd.read_csv(filepath, encoding="utf-8-sig")
            df = pd.concat([df, df_tmp], axis=0)
        os.makedirs(f"./filtered/kma/{year}")
        save_path = f"./filtered/kma/{year}/kma_{year}_{city}.csv"
        df.to_csv(save_path, index=False, encoding="utf-8-sig")