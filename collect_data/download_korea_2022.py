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
        df = pd.read_fwf(data_io, skiprows=4, skipfooter=1, header=None)
        

        # 첫 번째 행을 열 이름으로 사용 (열 이름을 수동으로 지정할 수도 있음)

        # After loading your data with `pd.read_fwf` or similar method
        df.columns = columns

        return df
    else:
        print(f"Error: {response.status_code}")

def save_csv(data_type, BASE_URL, API_KEY, columns):
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
WEATHER_BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
# year = 2018
years = [2018, 2019, 2020, 2021, 2022]

# stns = [184, 101, 108, 133, 136, 146, 143, 156, 159]    # 지점 번호
# locations = {184: "jeju", 101: "chuncheon", 133: "daejeon", 136: "andong",
#              146: "jeonju", 143: "daegu", 156: "gwangju", 159: "busan", 108: "seoul"}
stns = [136, 146, 143, 156, 108]    # 지점 번호
locations = {136: "andong", 146: "jeonju", 143: "daegu", 156: "gwangju", 108: "seoul"}

combinations = [f"{year}{month:02d}010000" for month in range(1, 13)]

weather_columns = [
        "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR",
        "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3",
        "SD_DAY", "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN",
        "CT", "CT_TOP", "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01",
        "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR", "IX"
        ]
for year in years:
    combinations = [f"{year}{month:02d}010000" for month in range(1, 13)]
    save_csv("weather", WEATHER_BASE_URL, API_KEY, weather_columns)