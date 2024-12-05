import pandas as pd
from datetime import timedelta
import logging
import torch
import torch.nn as nn
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

def concat_china_data(df):
    """
    중국 데이터 파일을 읽어와서 기존 DataFrame과 병합하는 함수
    """
    cma_locations = ["yanan", "tongliao", "qingdao", "chifeng", "dalian"]

    for loc in cma_locations:
        cma_path = f"../../collect_data/filtered/cma/{loc}_meta.csv"
        df_cma = pd.read_csv(cma_path)
        df_cma = df_cma.loc[:, ~df_cma.columns.str.contains("IW|IR|IX|RH|LON|LAT")]

        df = pd.merge(df, df_cma, on="TM", how="inner", suffixes=("", f"_{loc}"))

    df = df.drop_duplicates(subset='TM').reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains("ORG")]
    df = df.sort_values(by='TM').reset_index(drop=True)

    return df

def print_missing_info(df):
    """
    DataFrame의 shape과 결측치 비율을 출력하는 함수
    """
    # 결측치 비율 계산
    missing_ratio = df.isnull().mean() * 100

    # 0이 아닌 비율을 가진 열만 선택
    non_zero_missing = missing_ratio[missing_ratio > 0]

    # 결과 출력
    print(df.shape)
    print(non_zero_missing)
    print(df.columns)

def convert_timesteps(data):
    """
    datetime 형식으로 변환
    """
    # diff 열 계산
    data['TM'] = pd.to_datetime(data['TM'], format='%Y%m%d%H%M')
    data['diff'] = data['TM'].diff().dt.total_seconds() // 3600

    # TM 기준으로 정렬
    data = data.sort_values(by='TM').reset_index(drop=True)

    # diff 재계산
    data['diff'] = data['TM'].diff().dt.total_seconds() // 3600
    data.loc[0, 'diff'] = 3

    return data

def set_logger(filename):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        filename=filename,
        format='%(asctime)s - %(message)s',
        level=logging.DEBUG,
        force=True,
    )

    file_handler = logging.FileHandler(filename, mode='w')  # Overwrite mode
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    # Add the file handler to the root logger
    logging.getLogger().addHandler(file_handler)

def filter_columns(df, columns_to_remove):
    return df.drop(columns=columns_to_remove)

def prepare_data(region, columns_to_remove, include_china=False):
    filename = f"../../collect_data/filtered/kma/merged/kma_{region}_meta.csv"
    df = pd.read_csv(filename)
    if include_china:
        df = concat_china_data(df)
    df = filter_columns(df, columns_to_remove)
    df = convert_timesteps(df)
    return df