import os
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import time as _time
import chardet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import statsmodels.api as sm
from numpy.linalg import lstsq
from dateutil.relativedelta import relativedelta
import mplfinance as mpf
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import ttest_1samp
from arch import arch_model
import statsmodels.formula.api as smf
from datetime import time
import random

##匯入資料
etf_data = pd.read_csv("D:/我才不要走量化/ETF_overnight/nav_etf/etf.csv", encoding = 'utf-16',sep="\t")

file_path = r"D:\我才不要走量化\ETF_overnight\nav_etf\etf_nav.xlsx"

sheets_dict = pd.read_excel(file_path, sheet_name=None)

df_list = []

for sheet_name, df in sheets_dict.items():
   
    df["code"] = sheet_name
    

    df_list.append(df)


nav_data = pd.concat(df_list, ignore_index=True)

print(nav_data)

etf_data[["code", "name"]] = etf_data["證券代碼"].str.split(" ", n=1, expand=True)

rename_dict = {
    "年月日": "date",
    "開盤價(元)": "open",
    "最高價(元)": "high",
    "最低價(元)": "low",
    "收盤價(元)": "close",
    "成交量(千股)": "volume_k",
    "市值(百萬元)": "mkt_value_m",
    "證券代碼": "full_code"
}


etf_data = etf_data.rename(columns=rename_dict)

print(etf_data) #etf data is stock data

#######計算overnight######
#成分股
nav_data["date"] = pd.to_datetime(nav_data["date"], format="%Y/%m/%d", errors="coerce")
nav_data["date"] = nav_data["date"].dt.strftime("%Y-%m-%d")
nav_data = nav_data.sort_values(by=["code", "date"])
nav_data["prev_close"] = nav_data.groupby("code")["close"].shift(1)
nav_data["overnight_return_stock"] = (nav_data["open"] - nav_data["prev_close"]) / nav_data["prev_close"]

#市價
etf_data["date"] = pd.to_datetime(etf_data["date"], format="%Y%m%d", errors="coerce")
etf_data = etf_data.sort_values(by=["code", "date"])

etf_data["prev_close"] = etf_data.groupby("code")["close"].shift(1)

etf_data["overnight_return"] = (etf_data["open"] - etf_data["prev_close"]) / etf_data["prev_close"]

print(etf_data[["code", "name", "date", "open", "close", "prev_close", "overnight_return"]].head(10))

####合併####
etf_data["date"] = pd.to_datetime(etf_data["date"], errors="coerce")
nav_data["date"] = pd.to_datetime(nav_data["date"], errors="coerce")

nav_merge = nav_data[["date", "code", "overnight_return_stock"]].copy()

etf_data_merged = pd.merge(
    etf_data,
    nav_merge,
    on=["date", "code"],
    how="left",             
    validate="m:1"          
)

###regression###
reg_data = etf_data_merged.dropna(subset=["overnight_return", "overnight_return_stock"]).copy()

y = reg_data["overnight_return"]
x = reg_data["overnight_return_stock"]

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()

print(model.summary())

####2021~2023y####
reg_data_2021_2023 = reg_data[(reg_data["date"] >= "2021-01-01") & (reg_data["date"] <= "2023-12-31")].copy()
y_2021_2023 = reg_data_2021_2023["overnight_return"]
x_2021_2023 = reg_data_2021_2023["overnight_return_stock"]
x_2021_2023 = sm.add_constant(x_2021_2023)
model_2021_2023 = sm.OLS(y_2021_2023, x_2021_2023).fit()
print(model_2021_2023.summary())

#### 2024~now ####
reg_data_2024_now = reg_data[reg_data["date"] >= "2024-01-01"].copy()
y_2024_now = reg_data_2024_now["overnight_return"]
x_2024_now = reg_data_2024_now["overnight_return_stock"]
x_2024_now = sm.add_constant(x_2024_now)
model_2024_now = sm.OLS(y_2024_now, x_2024_now).fit()
print(model_2024_now.summary())


#2021至2023的偏誤可能還明顯一點點，但我猜確實隨著etf越來越成熟
#喔喔喔喔!!!我忘記截距項作者有轉成bps!!!!只要是顯著的都可以再做看看!!
#真的是4bps 哇!!! beta:etf overnight 漲1%，成分股overnight 漲 0.89%

print("2021-2023 Intercept (bps):", model_2021_2023.params['const'] * 10000)
print("2024-now Intercept (bps):", model_2024_now.params['const'] * 10000)