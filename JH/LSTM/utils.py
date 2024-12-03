import pandas as pd
from datetime import timedelta


def concat_china_data(df):
    """
    중국 데이터 파일을 읽어와서 기존 DataFrame과 병합하는 함수
    """
    cma_locations = ["yanan", "tongliao", "qingdao", "chifeng", "dalian"]

    for loc in cma_locations:
        cma_path = f"../../collect_data/filtered/cma/{loc}_meta.csv"
        df_cma = pd.read_csv(cma_path)

        df = pd.merge(df, df_cma, on="TM", how="inner", suffixes=("", f"_{loc}"))

    df = df.drop_duplicates(subset='TM').reset_index(drop=True)
    return df

def print_missing_info(df):
    """
    DataFrame의 shape과 결측치 비율을 출력하는 함수
    """
    print(df.shape)

    # 결측치 비율 계산
    missing_ratio = df.isnull().mean() * 100

    # 0이 아닌 비율을 가진 열만 선택
    non_zero_missing = missing_ratio[missing_ratio > 0]

    # 결과 출력
    print(non_zero_missing)

    print(df.columns)

def convert_timesteps(data):
    """
    datetime 형식으로 변환 후 6시간 차이 날 경우 데이터 보간
    """
    # diff 열 계산
    data['TM'] = pd.to_datetime(data['TM'], format='%Y%m%d%H%M')
    data['diff'] = data['TM'].diff().dt.total_seconds() // 3600

    # 새로운 행을 저장할 리스트
    new_rows = []

    for i in range(1, len(data)):
        if data.loc[i, 'diff'] == 6.0:  # 6시간 간격인 경우
            # 이전 행 가져오기
            prev_row = data.iloc[i - 1].copy()
            # 새로운 행 삽입 (3시간 간격)
            new_row = prev_row.copy()
            new_row['TM'] += timedelta(hours=3)
            new_rows.append(new_row)

    # 기존 데이터프레임에 새로운 행 추가
    data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True)
    # TM 기준으로 정렬
    data = data.sort_values(by='TM').reset_index(drop=True)

    # diff 재계산
    data['diff'] = data['TM'].diff().dt.total_seconds() // 3600
    data.iloc[0, 'diff'] = 3

    return data

