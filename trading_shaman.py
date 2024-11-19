import base64
import functions_framework
import os
# import time
import pandas as pd
import numpy as np
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import pandas_ta as ta
import requests
import yaml

def compute_gradient(start_index, df, x_label, y_label, len_data):
    # Ensure we only take data points from n to n+5
    if start_index + len_data > len(df):
        return None  # Return None if there are not enough points to calculate gradient

    # Extract the last 5 data points (x, y) from the DataFrame
    data_segment = df.iloc[start_index:start_index + len_data]
    x = data_segment[x_label]
    y = data_segment[y_label]

    # Calculate the necessary summations for the least squares formula
    n = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_x2 = (x ** 2).sum()
    sum_xy = (x * y).sum()

    # Calculate the slope (gradient) using the least squares formula
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    return slope

def compute_gradient_normalize(start_index, df, x_label, y_label, len_data):
    # Ensure we only take data points from n to n+5
    if start_index + len_data > len(df):
        return None  # Return None if there are not enough points to calculate gradient

    # Extract the last 5 data points (x, y) from the DataFrame
    data_segment = df.iloc[start_index:start_index + len_data]
    x = data_segment[x_label]
    y = data_segment[y_label]

    #normalize y
    min_y = np.min(y)
    max_y = np.max(y)
    y = (y - min_y) / (max_y - min_y)

    # Calculate the necessary summations for the least squares formula
    n = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_x2 = (x ** 2).sum()
    sum_xy = (x * y).sum()

    # Calculate the slope (gradient) using the least squares formula
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    return slope

def check_crossing(df, col1, col2):
    # Calculate the difference between the two columns
    diff = df[col1] - df[col2]
    diff = diff / np.abs(diff)
    # Check if there is a sign change in the difference
    crossing = ((diff.shift(1) * diff) - 1) / -2
    return crossing

class Crypto:
    def __init__(self, ticker='ETH', market='USD', key=''):
        self.ticker = ticker
        self.market = market
        self.key = key

    def rename_column(self, df):
        df = df.rename(columns={"4. close": "close",
                                "1. open": "open",
                                "2. high": "high",
                                "3. low": "low",
                                "5. volume": "volume"})

        df = df.sort_index()
        df['idx_int'] = np.arange(0, len(df))
        df = df.reset_index()

        return df[['date', 'idx_int', 'open', 'high', 'low', 'close', 'volume']]

    def get_intraday(self):
        cc = CryptoCurrencies(key=self.key, output_format='pandas')
        df, self.meta_data = cc.get_crypto_intraday(symbol=self.ticker, market=self.market, interval='1min', outputsize='full')
        return df

    def get_technical_indicators(self, df):
        df.ta.ema(length=10, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.psar(append=True)
        df.ta.adx(append=True)
        return df

    def get_misc_indicators(self, df, len_data=3, len_data_norm=15):
        for i in range(len(df) - len_data):  # Make sure we have at least 5 points for each calculation
            gradient = compute_gradient(i, df, 'idx_int', 'EMA_10', len_data)
            df.at[i + len_data, 'gradient_ema_10'] = gradient  # Store the gradient in the row corresponding to n+4
            gradient = compute_gradient(i, df, 'idx_int', 'RSI_14', len_data)
            df.at[i + len_data, 'gradient_rsi_14'] = gradient  # Store the gradient in the row corresponding to n+4
            gradient = compute_gradient(i, df, 'idx_int', 'close', len_data)
            df.at[i + len_data, 'gradient_ls'] = gradient
            gradient = compute_gradient(i, df, 'idx_int', 'EMA_50', len_data)
            df.at[i + len_data, 'gradient_ema_50'] = gradient

        for i in range(len(df) - len_data_norm):  # Make sure we have at least 15 points for each calculation
            gradient = compute_gradient_normalize(i, df, 'idx_int', 'EMA_10', len_data_norm)
            df.at[i + len_data_norm, 'gradient_norm_ema_10'] = gradient  # Store the gradient in the row corresponding to n+4

        df['r_ema_s_m'] = df['EMA_10'] / df['EMA_50']
        df['flag_ema_crossing'] = check_crossing(df, 'EMA_10', 'EMA_50')

        df['psar_flip_dir'] = 0
        df.loc[(df['PSARr_0.02_0.2'] == 1) & (df['PSARl_0.02_0.2'].isnull() == False), 'psar_flip_dir'] = 1
        df.loc[(df['PSARr_0.02_0.2'] == 1) & (df['PSARs_0.02_0.2'].isnull() == False), 'psar_flip_dir'] = -1

        mask_ema_grad_pos = (df['gradient_ema_10'] > 0.05)
        mask_ema_grad_neg = (df['gradient_ema_10'] < -0.05)
        df['flag_grad_ema'] = 0
        df.loc[mask_ema_grad_pos, 'flag_grad_ema'] = 1
        df.loc[mask_ema_grad_neg, 'flag_grad_ema'] = -1

        mask_ema_grad_pos = (df['gradient_ema_50'] > 0.05)
        mask_ema_grad_neg = (df['gradient_ema_50'] < -0.05)
        df['flag_grad_ema_50'] = 0
        df.loc[mask_ema_grad_pos, 'flag_grad_ema_50'] = 1
        df.loc[mask_ema_grad_neg, 'flag_grad_ema_50'] = -1

        mask_rsi_grad_pos = (df['gradient_rsi_14'] >= 1)
        mask_rsi_grad_neg = (df['gradient_rsi_14'] <= 1)
        df['flag_grad_rsi'] = 0
        df.loc[mask_rsi_grad_pos, 'flag_grad_rsi'] = 1
        df.loc[mask_rsi_grad_neg, 'flag_grad_rsi'] = -1

        df['flag_grad_ls'] = 0
        df.loc[df['gradient_ls'] >= 0.05, 'flag_grad_ls'] = 1
        df.loc[df['gradient_ls'] <= -0.05, 'flag_grad_ls'] = -1

        df['ema_short_above_or_below'] = 0
        df.loc[(df['EMA_10'] > df['EMA_50']), 'ema_short_above_or_below'] = 1
        df.loc[(df['EMA_10'] < df['EMA_50']), 'ema_short_above_or_below'] = -1

        df['r_close_bbu'] = df['close'] / df['BBU_20_2.0']
        df['r_close_bbl'] = df['close'] / df['BBL_20_2.0']
        df['r_ema_bbu'] = df['EMA_10'] / df['BBU_20_2.0']
        df['r_ema_bbl'] = df['EMA_10'] / df['BBL_20_2.0']
        return df

    def create_signal(self, df):
        # Trend confirmation
        mask_bulber = (df['ADX_14'] >= 25)
        mask_bul = (df['DMP_14'] >= 25)
        mask_ber = (df['DMN_14'] >= 25)

        df['trend_confirm'] = 0
        df.loc[mask_bulber & mask_bul, 'trend_confirm'] = 1
        df.loc[mask_bulber & mask_ber, 'trend_confirm'] = -1

        # Buy Signal
        mask_le1 = (df['ema_short_above_or_below'] == 1) & (df['flag_ema_crossing'] == 1)
        mask_le2 = (df['MACDh_12_26_9'] > 0)
        mask_le3 = (df['r_close_bbl'] <= 1.0005)
        mask_le4 = (df['RSI_14'] < 70) & (df['RSI_14'] > 30)
        mask_le5 = (df['PSARl_0.02_0.2'] < df['close']) & (df['psar_flip_dir'] > 0)
        mask_le6 = (df['RSI_14'] < 40)
        mask_le7 = (df['flag_grad_ema'] >= 0)

        df['ema_crossing_pos'] = 0
        df.loc[mask_le1, 'ema_crossing_pos'] = 1
        df['macd_pos'] = 0
        df.loc[mask_le2, 'macd_pos'] = 1
        df['close_to_bbl'] = 0
        df.loc[mask_le3, 'close_to_bbl'] = 1
        df['rsi_30_to_70'] = 0
        df.loc[mask_le4, 'rsi_30_to_70'] = 1
        df['PSAR_bellow_close'] = 0
        df.loc[mask_le5, 'PSAR_bellow_close'] = 1

        df['buy_signal'] = np.nan
        # df.loc[(mask_le1 & mask_le4) | (mask_le5 & mask_le4 & mask_le2) | (mask_le2 & mask_le6 & mask_le3), 'long_entry'] = 1
        # df.loc[(mask_le1 & mask_le4) | (mask_le6 & mask_le7), 'buy_signal'] = 1
        df.loc[(mask_le1 & mask_le4) | (mask_le3 & mask_le7), 'buy_signal'] = 1

        # Sell Signal
        mask_lex1 = (df['ema_short_above_or_below'] == -1) & (df['flag_ema_crossing'] == 1)
        # mask_lex2 = (df['RSI_14']>70)
        mask_lex2 = (df['RSI_14'] > 55)
        mask_lex3 = (df['psar_flip_dir'] == -1)
        mask_lex4 = (df['flag_grad_ema'] == 0)
        mask_lex5 = (df['MACDh_12_26_9'] < 0)

        df['ema_crossing_neg'] = 0
        df.loc[mask_lex1, 'ema_crossing_neg'] = 1
        df['rsi_above_70'] = 0
        df.loc[mask_lex2, 'rsi_above_70'] = 1
        df['psar_flip_neg'] = 0
        df.loc[mask_lex3, 'psar_flip_neg'] = 1
        df['macd_neg'] = 0
        df.loc[mask_lex5, 'macd_neg'] = 1

        df['sell_signal'] = np.nan
        df.loc[(mask_lex1) | (mask_lex2 & mask_lex4), 'sell_signal'] = 1

        #Over-bought/Sold
        mask_os1 = (df['RSI_14'] <= 20)
        mask_os2 = (df['r_close_bbl'] <= 1.000)
        mask_ob1 = (df['RSI_14'] >= 80)
        mask_ob2 = (df['r_close_bbu'] >= 1.000)
        df['oversold_confirm'] = 0
        df.loc[mask_os1, 'oversold_confirm'] = 1
        df.loc[mask_ob1, 'oversold_confirm'] = -1

        cols_use = ['oversold_confirm', 'trend_confirm', 'sell_signal', 'buy_signal', 'flag_ema_crossing',
                    'ema_short_above_or_below', 'flag_grad_ema', 'gradient_norm_ema_10', 'RSI_14', 'EMA_10', 'EMA_50', 'ADX_14', 'DMP_14',
                    'DMN_14', 'MACDh_12_26_9', 'close', 'BBU_20_2.0', 'BBL_20_2.0']

        return df.iloc[-1:][cols_use].sum()

def send_message(token, chat_id, msg=""):
    TOKEN = token
    chat_id = chat_id
    message = msg
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json())  # this sends the message

def get_crypto_signal(key, ticker='ETH', market='USD'):
    eth = Crypto(ticker=ticker, market=market, key=key)
    df = eth.get_intraday()
    df = eth.rename_column(df)
    df = eth.get_technical_indicators(df)
    df = eth.get_misc_indicators(df)
    df = eth.create_signal(df)
    return df

def execute_command(key, ticker, TOKEN, chat_id):
    # Code to be measured
    series_crp = get_crypto_signal(key, ticker=ticker, market='USD')
    send_str = f"\n\n{ticker}:\n{series_crp}"
    if (series_crp.loc[["sell_signal", "buy_signal"]].sum() > 0) or (series_crp.loc["oversold_confirm"].sum() != 0):
        send_message(TOKEN, chat_id, send_str)

def hello_shaman(config):
    key = config['av_key']
    TOKEN = config['telegram_bot_key']

    chat_id = config['eth_chat_id']
    ticker = 'ETH'
    execute_command(key, ticker, TOKEN, chat_id)

    chat_id = config['btc_chat_id']
    ticker = 'BTC'
    execute_command(key, ticker, TOKEN, chat_id)

if __name__ == "__main__":
  config = dict()
  config['av_key'] = os.getenv('AV_KEY')
  config['telegram_bot_key'] = os.getenv('TEL_BOT_KEY')
  config['eth_chat_id'] = os.getenv('ETH_CHAT_ID')
  config['btch_chat_id'] = os.getenv('BTC_CHAT_ID')

  hello_shaman(config)
