import requests
import pandas as pd
from io import StringIO
from itertools import product
import os
from tqdm import tqdm

def generate_combinations(start_year, end_year):
    """
    지정된 연도 범위에 따라 월별 시작 날짜와 종료 날짜의 조합을 생성합니다.
    """
    combinations = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            combinations.append(f"{year}{month:02d}010000")
    # 종료 날짜 추가
    combinations.append(f"{end_year + 1}01010000")
    return combinations


def api_to_dataframe(url, columns):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.content
        decoded_data = data.decode('euc-kr')
        
        data_io = StringIO(decoded_data)    # 문자열로 이미 제공되었기 때문에 StringIO를 사용해 읽음
        # 고정 너비의 데이터를 읽어옵니다.
        df = pd.read_csv(data_io, skiprows=16, header=None, names=columns, engine='c')
        df = df.iloc[:-1]
        df['TM'] = df['TM'].astype(str)

        return df
    else:
        print(f"Error: {response.status_code}")

def save_csv(data_type, base_url, api_key, columns, stns, locations, start_year, end_year):
    ### Create folder
    data_dir = os.path.join("data", "cma", data_type)
    os.makedirs(data_dir, exist_ok=True)

    combinations = generate_combinations(start_year, end_year)

    for stn in tqdm(stns, desc="Stations"):
        df = pd.DataFrame()
        location_name = locations.get(stn, f"station_{stn}")
        save_path = os.path.join(data_dir, f"cma_{start_year}_{end_year}_{location_name}.csv")

        for i in tqdm(range(len(combinations) - 1), desc=f"{location_name} 데이터 수집 중", leave=False):
            TM1 = combinations[i]
            TM2 = combinations[i+1]

            url = f"{base_url}?tm_st={TM1}&tm={TM2}&stn={stn}&org='cma'&data=1&mode=1&help=0&authKey={api_key}"
            
            df_tmp = api_to_dataframe(url, columns)
            if not df_tmp.empty:
                df = pd.concat([df, df_tmp], axis=0, ignore_index=True)
        
        if not df.empty:
            df.to_csv(save_path, index=False)
            print(f"{location_name} 데이터가 {save_path}에 저장되었습니다.")
        else:
            print(f"{location_name}에 대한 데이터를 가져오지 못했습니다.")


def main():
    ### 지상 관측자료 (시간)
    API_KEY = "n82QCM3KRbCNkAjNyoWwTg"
    FINEDUST_BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/dst_pm10_tm.php"

    stns = [54857, 54662, 54135, 53845, 54218]    # 지점 번호
    locations = {
        54857: "qingdao",
        54662: "dalian",
        54135: "tongliao",
        53845: "yanan",
        54218: "chifeng"
    }

    start_year = 2018
    end_year = 2022

    finedust_columns = ["TM", "ORG", "STN", "PM10"]

    save_csv("finedust", FINEDUST_BASE_URL, API_KEY, finedust_columns, stns, locations, start_year, end_year)


if __name__ == "__main__":
    main()