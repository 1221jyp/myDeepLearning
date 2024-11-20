import pandas as pd
import yfinance as yf
import os
print(os.getcwd())


# 비트코인 가격 데이터 수집
btc_data = yf.download('BTC-KRW', start='2020-01-01', end='2024-11-16')
btc_data.to_csv('btc_data.csv')

