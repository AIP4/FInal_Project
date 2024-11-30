import os
import pandas as pd
from tqdm import tqdm

years = [2018, 2019, 2020, 2021, 2022]
locations = ["chifeng", "dalian", "qingdao", "tongliao", "yanan"]
target_dir = "filteerd/kma_3H"

for location in locations:
    os.makedirs("filtered/cma_1H", exist_ok=True)
    file_path = f"filtered/cma/{location}.csv"

    data = pd.read_csv(file_path)

    data["TM"] = pd.to_datetime(data["TM"], format='%Y%m%d%H%M')

    data = data.set_index("TM").resample("1H").ffill().reset_index()
    data["TM"] = data["TM"].dt.strftime('%Y%m%d%H%M')

    data.to_csv(f"filtered/cma_1H/{location}.csv", index=False)
