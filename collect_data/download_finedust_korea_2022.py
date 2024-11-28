import requests
import pandas as pd
from io import StringIO
from itertools import product
import os
from tqdm import tqdm

def api_to_csv(url, save_path, columns):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.content
        decoded_data = data.decode('euc-kr')
        
        data_io = StringIO(decoded_data)    # 문자열로 이미 제공되었기 때문에 StringIO를 사용해 읽음
        # 고정 너비의 데이터를 읽어옵니다.
        df = pd.read_csv(data_io, skiprows=12, header=None, names=columns, engine='c')
        df = df.iloc[:-1]
        df['TM'] = df['TM'].astype(str)

        # 시간 단위 필터링
        df_hourly = df[df['TM'].str.endswith('00')]
        df_hourly = df_hourly.iloc[:, :3]

        return df_hourly
    else:
        print(f"Error: {response.status_code}")

def save_csv(data_type, BASE_URL, API_KEY, columns, year):
    ### Create folder
    os.makedirs(f"data/kma/{year}/{data_type}", exist_ok=True)

    for stn in tqdm(stns, desc="Stations"):
        df = pd.DataFrame()
        SAVE_PATH = f"./data/kma/{year}/{data_type}/kma_{year}_{locations[stn]}.csv"

        for i in tqdm(range(len(combinations) - 1), desc=f"Processing combinations for {locations[stn]}", leave=False):
            TM1 = combinations[i]
            TM2 = combinations[i+1]

            url = f"{BASE_URL}?tm1={TM1}&tm2={TM2}&stn={stn}&help=0&authKey={API_KEY}"
            
            df_tmp = api_to_csv(url, SAVE_PATH, columns)
            df = pd.concat([df, df_tmp], axis=0)
        df.to_csv(SAVE_PATH, index=False)



### 지상 관측자료 (시간)
API_KEY = "n82QCM3KRbCNkAjNyoWwTg"
FINEDUST_BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_pm10.php"
# year = 2018
years = [2018, 2019, 2020, 2021, 2022]

stns = [136, 146, 143, 156, 108]    # 지점 번호
locations = {136: "andong", 146: "jeonju", 143: "daegu", 156: "gwangju", 108: "seoul"}

combinations = [f"{year}{month:02d}010000" for month in range(1, 13)]

finedust_columns = [
    "TM", "STN_ID", "PM10", "NULL1", "NULL2", "NULL3"
]

for year in years:
    combinations = [f"{year}{month:02d}010000" for month in range(1, 13)]
    save_csv("finedust", FINEDUST_BASE_URL, API_KEY, finedust_columns, year)