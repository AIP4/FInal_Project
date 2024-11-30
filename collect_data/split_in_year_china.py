import os
import pandas as pd
from tqdm import tqdm

years = [2018, 2019, 2020, 2021, 2022]
locations = ["chifeng", "dalian", "qingdao", "tongliao", "yanan"]
target_dir = "filteerd/kma_3H"

for location in locations:
    # os.makedirs(f"filtered/cma_1H/{year}/", exist_ok=True)
    file_path = f"filtered/cma_1H/{location}.csv"

    data = pd.read_csv(file_path)
    data["TM"] = pd.to_datetime(data["TM"], format='%Y%m%d%H%M')

    for year in years:
        os.makedirs(f"filtered/cma_1H/{year}", exist_ok=True)
        yearly_data = data[data["TM"].dt.year == year]
        output_file = f"filtered/cma_1H/{year}/{location}.csv"

        yearly_data['TM'] = pd.to_datetime(yearly_data['TM']).dt.strftime('%Y%m%d%H%M')
        yearly_data.to_csv(output_file, index=False)



