import requests
import pandas as pd
from io import StringIO
from itertools import product
import os

def api_to_csv(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.content
        decoded_data = data.decode('euc-kr')
        
        data_io = StringIO(decoded_data)    # 문자열로 이미 제공되었기 때문에 StringIO를 사용해 읽음
        # 고정 너비의 데이터를 읽어옵니다.
        df = pd.read_fwf(data_io, skiprows=4, skipfooter=1, header=None)

        # 첫 번째 행을 열 이름으로 사용 (열 이름을 수동으로 지정할 수도 있음)
        columns = [
        "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR",
        "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3",
        "SD_DAY", "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN",
        "CT", "CT_TOP", "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01",
        "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR", "IX"
        ]

        # After loading your data with `pd.read_fwf` or similar method
        df.columns = columns

        df.to_csv(save_path, index=False, encoding="utf-8-sig")
    else:
        print(f"Error: {response.status_code}")

def create_year_month_combinations(years, months):
    return [f"{year}{month}" for year, month in product(years, months)]



### Create folder
os.makedirs("data/kma", exist_ok=True)

### 지상 관측자료 (시간)
API_KEY = "n82QCM3KRbCNkAjNyoWwTg"

years = ["2018", "2019", "2020", "2021", "2022"]
months = [f"{month:02d}01" for month in range(1, 13)]
combinations = create_year_month_combinations(years, months)
for i in range(len(combinations)-1):
    TM1 = combinations[i] + "0100"
    TM2 = combinations[i+1] + "0000"

    BASE_URL = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php?tm1={TM1}&tm2={TM2}&stn=108&help=0&authKey={API_KEY}"
    SAVE_PATH = "./data/kma/" + combinations[i] + ".csv"
    api_to_csv(BASE_URL, SAVE_PATH)
