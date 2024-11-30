import os
import pandas as pd
from tqdm import tqdm

years = [2018, 2019, 2020, 2021, 2022]
locations = ["andong", "daegu", "gwangju", "jeonju", "seoul"]
target_dir = "filteerd/kma_3H"

columns = ['TM', 'STN_ID', 'PM10', 'STN', 'WD', 'WS', 'GST_WD', 
           'GST_WS', 'GST_TM', 'PA', 'PS', 'PT', 'PR', 'TA', 'TD', 
           'HM', 'PV', 'RN', 'RN_DAY', 'RN_JUN', 'RN_INT', 'SD_HR3', 
           'SD_DAY', 'SD_TOT', 'WC', 'WP', 'WW', 'CA_TOT', 'CA_MID', 
           'CH_MIN', 'CT', 'CT_TOP', 'CT_MID', 'CT_LOW', 'VS', 'SS', 
           'SI', 'ST_GD', 'TS', 'TE_005', 'TE_01', 'TE_02', 'TE_03', 
           'ST_SEA', 'WH', 'BF', 'IR', 'IX']

extracted_columns = ['TM']

for year in tqdm(years):
    for location in tqdm(locations):
        os.makedirs(f"filtered/kma_reduced_dimension/{year}/", exist_ok=True)
        file_path = f"filtered/kma/{year}/kma_{year}_{location}.csv"
        data = pd.read_csv(file_path)

        resampled_data = data[extracted_columns]

        resampled_data.to_csv(f'filtered/kma_reduced_dimension/{year}/kma_{year}_{location}.csv', index=False)