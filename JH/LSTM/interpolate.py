import pandas as pd
from datetime import timedelta

def fill_intervals(data):
    """
    3시간 초과, 일주일 이내인 경우만 보간
    """
    # 새로운 행을 저장할 리스트
    new_rows = []

    for i in range(1, len(data)):
        diff = data.loc[i, 'diff']
        if diff > 3.0 and diff <= 168.0:  # 3시간 초과, 168시간 이하인 경우만 보간
            # 이전 행과 현재 행 가져오기
            prev_row = data.iloc[i - 1]
            current_row = data.iloc[i]

            # 필요한 중간 행 생성
            num_new_rows = int(diff // 3) - 1  # 필요한 중간 행의 개수
            for j in range(1, num_new_rows + 1):
                mid_row = prev_row.copy()
                mid_row['TM'] += timedelta(hours=3 * j)  # 3시간씩 추가
                new_rows.append(mid_row)

    # 기존 데이터프레임에 새로운 행 추가
    data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True)

    # TM 기준으로 정렬
    data = data.sort_values(by='TM').reset_index(drop=True)

    return data


def simple_interpolate(data, method="linear"):
    """
    기본적인 보간법 (method="linear" or "time")
    """
    data = fill_intervals(data)
    # 데이터 보간 (168 초과 간격은 제외되므로 안전)
    if method == "linear":
        data = data.interpolate(method='linear')
    elif method == "time":
        data = data.interpolate(method='time')

    # diff 재계산
    data['diff'] = data['TM'].diff().dt.total_seconds() // 3600

    return data