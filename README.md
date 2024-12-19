# 24-2 AI Project : Air Quality Prediction
---
본 프로젝트는 다양한 딥러닝 모델링을 활용하여 미세먼지 농도(PM10)을 예측하고, 각 특성을 잘 반영할 수 있는 XAI 방법을 제시하는 것을 목표로 합니다.

## Team
---
- Members : 신재환, 한준호, 김태영, 유재룡, 하상수

## Roles
---
- 신재환
    - Baselien Modeling (MLP)
    - Experiment Design
    - Analysis
- 한준호
    - Data Collection
    - Data Preprocessing
    - Experiment Design
    - Modeling (LSTM, 1DCNN-LSTM)
    - Analysis
    - Code Refactoring
    - PPT
- 김태영
    - Data Collection
    - Data Preprocessing
    - PPT
- 유재룡
    - PPT
- 하상수
    - Report

## Weather Data Columns
---
이 데이터셋은 다양한 기상 데이터를 포함하고 있습니다. 각 컬럼의 설명은 아래와 같습니다:

| Column Name | Description |
|-------------|-------------|
| **TM**      | Timestamp of the data (Format: YYYYMMDDHHMM). |
| **STN**     | Station ID where the data was collected. |
| **WD**      | Wind direction (in degrees). |
| **WS**      | Wind speed (in meters per second). |
| **GST_WD**  | Gust wind direction (in degrees). |
| **GST_WS**  | Gust wind speed (in meters per second). |
| **GST_TM**  | Timestamp of the strongest gust. |
| **PA**      | Atmospheric pressure at station level (in hPa). |
| **PS**      | Atmospheric pressure at sea level (in hPa). |
| **PT**      | Precipitation type (e.g., rain, snow). |
| **PR**      | Precipitation amount (in millimeters). |
| **TA**      | Air temperature (in degrees Celsius). |
| **TD**      | Dew point temperature (in degrees Celsius). |
| **HM**      | Humidity (in percentage). |
| **PV**      | Vapor pressure (in hPa). |
| **RN**      | Rainfall amount (in millimeters). |
| **RN_DAY**  | Daily rainfall amount (in millimeters). |
| **RN_JUN**  | Cumulative rainfall amount (in millimeters). |
| **RN_INT**  | Rainfall intensity (in millimeters per hour). |
| **SD_HR3**  | Snow depth (3-hour average, in centimeters). |
| **SD_DAY**  | Daily snow depth (in centimeters). |
| **SD_TOT**  | Total snow depth (in centimeters). |
| **WC**      | Weather code (e.g., clear, cloudy, rain). |
| **WP**      | Weather phenomenon code (e.g., fog, thunderstorm). |
| **WW**      | Current weather condition (short description). |
| **CA_TOT**  | Total cloud amount (in octas). |
| **CA_MID**  | Middle-level cloud amount (in octas). |
| **CH_MIN**  | Minimum cloud height (in meters). |
| **CT**      | Cloud type (e.g., cumulus, stratus). |
| **CT_TOP**  | Top height of clouds (in meters). |
| **CT_MID**  | Middle height of clouds (in meters). |
| **CT_LOW**  | Bottom height of clouds (in meters). |
| **VS**      | Visibility (in kilometers). |
| **SS**      | Sunshine duration (in hours). |
| **SI**      | Solar radiation intensity (in W/m²). |
| **ST_GD**   | Soil temperature at ground depth (in degrees Celsius). |
| **TS**      | Ground surface temperature (in degrees Celsius). |
| **TE_005**  | Temperature at 0.05m depth (in degrees Celsius). |
| **TE_01**   | Temperature at 0.1m depth (in degrees Celsius). |
| **TE_02**   | Temperature at 0.2m depth (in degrees Celsius). |
| **TE_03**   | Temperature at 0.3m depth (in degrees Celsius). |
| **ST_SEA**  | Sea surface temperature (in degrees Celsius). |
| **WH**      | Wave height (in meters). |
| **BF**      | Beaufort scale of wind force. |
| **IR**      | Infrared radiation (in W/m²). |
| **IX**      | Index value (custom index specific to dataset). |
