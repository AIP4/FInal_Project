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
                
        print(f"concating: {sub_csv_files}")
        df1 = pd.read_csv(sub_csv_files[0])
        df2 = pd.read_csv(sub_csv_files[1])

        os.makedirs(f"./filtered/kma/{year}", exist_ok=True)

        df = pd.merge(df1, df2, on="TM", how="inner")

        save_path = f"./filtered/kma/{year}/kma_{year}_{city}.csv"
        df.to_csv(save_path, index=False, encoding="utf-8-sig")