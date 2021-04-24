# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Essentials
import gc
import numpy as np
import pandas as pd
import datetime
import random
import warnings
import string
from skopt.space import Real, Categorical, Integer
warnings.filterwarnings("ignore")
import functools
import dask
import os
CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])

# Plots
import matplotlib.pyplot as plt
# -

# # 1 数据处理

# ## 1.1 读取数据

data = pd.read_csv("20210104.csv")
data

data.shape

IC = data[data["InstrumentID"] == "IC2101"].reset_index(drop = True)

IC

#time = IC.iloc[:,2]
#orderbook = IC.iloc[:,10:30]
#IC = pd.concat([time, orderbook], axis = 1)
IC = IC.iloc[1:27425,7:27].reset_index(drop = True)
IC

IC = IC.rename(columns={"\t\t\t\t\tBidPrice1":"BidPrice1",
                       "\t\t\t\t\tBidPrice2":"BidPrice2",
                       "\t\t\t\t\tBidPrice3":"BidPrice3",
                       "\t\t\t\t\tBidPrice4":"BidPrice4",
                       "\t\t\t\t\tBidPrice5":"BidPrice5",
                       "AskVolume1":"AskVol1",
                        "AskVolume2":"AskVol2",
                       "AskVolume3":"AskVol3",
                       "AskVolume4":"AskVol4",
                       "AskVolume5":"AskVol5",
                       "BidVolume1":"BidVol1",
                        "BidVolume2":"BidVol2",
                       "BidVolume3":"BidVol3",
                       "BidVolume4":"BidVol4",
                       "BidVolume5":"BidVol5"})
IC

features = IC.copy()

# # 2 特征生成

# ## 2.1 Basic Set

# +
#V1 set Price and Volumn(5 levels)
#for i in range(5):    
#    features['AskPrice'+ str(i+1)] = features['AskPrice'+ str(i+1)]/10000
#    features['BidPrice'+ str(i+1)] = features['BidPrice'+ str(i+1)]/10000
# -

features

# ## 2.2 Time-insensitive Set

#V2 set Bid-ask spreads and mid-prices
for i in range(5):    
    features['Bid_ask_spread'+ str(i+1)] = features['AskPrice'+ str(i+1)] - features['BidPrice'+ str(i+1)]
    features['Mid_price'+ str(i+1)] = (features['BidPrice'+ str(i+1)]+ features['AskPrice'+ str(i+1)])/2

#V3 set Price differences
for i in range(4):    
    features['price_range_ask'+ str(i+2)] = features['AskPrice'+ str(i+2)] - features['AskPrice'+ str(i+1)]
    features['price_range_bid'+ str(i+2)] = features['BidPrice'+ str(i+2)] - features['BidPrice'+ str(i+1)]

#V4 set Mean prices and volumes   
features['mean_ask_price'] = (features['AskPrice1']+features['AskPrice2']+features['AskPrice3']+features['AskPrice4']+features['AskPrice5'])/5
features['mean_bid_price'] = (features['BidPrice1']+features['BidPrice2']+features['BidPrice3']+features['BidPrice4']+features['BidPrice5'])/5
features['mean_ask_volumn'] = (features['AskVol1']+features['AskVol2']+features['AskVol3']+features['AskVol4']+features['AskVol5'])/5
features['mean_bid_volumn'] = (features['BidVol1']+features['BidVol2']+features['BidVol3']+features['BidVol4']+features['BidVol5'])/5

#V5 set Accumulated differences
features['Accumulated_differences_price'] = (features['AskPrice1']-features['BidPrice1']+features['AskPrice2']-features['BidPrice2']+
                                             features['AskPrice3']-features['BidPrice3']+features['AskPrice4']-features['BidPrice4']+
                                             features['AskPrice5']-features['BidPrice5'])
features['Accumulated_differences_vol'] = (features['AskVol1']-features['BidVol1']+features['AskVol2']-features['BidVol2']+
                                             features['AskVol3']-features['BidVol3']+features['AskVol4']-features['BidVol4']+
                                             features['AskVol5']-features['BidVol5'])

# ## 2.3 Time-sensitive Set

#V6 Price and Volumn derivatives
for i in range(5):    
    features['Ask_price_derivative'+ str(i+1)] = (features['AskPrice'+ str(i+1)] - features['AskPrice'+ str(i+1)].shift(5))/5
    features['Bid_price_derivative'+ str(i+1)] = (features['BidPrice'+ str(i+1)] - features['BidPrice'+ str(i+1)].shift(5))/5
    features['Ask_vol_derivative'+ str(i+1)] = (features['AskVol'+ str(i+1)] - features['AskVol'+ str(i+1)].shift(5))/5
    features['Bid_vol_derivative'+ str(i+1)] = (features['BidVol'+ str(i+1)] - features['BidVol'+ str(i+1)].shift(5))/5

features = features.iloc[6:,:].reset_index(drop = True)

features

# ## 2.4 Label标注

# mid-price(upward,stationary,downward)


#mid-price
features['Mid_price'] = 0.5*(features['AskPrice1'] + features['BidPrice1'])

features['Mid_price_movement'] = np.zeros((len(features), 1))

# %%time
# set upward as 0, stationary as 1, downward as 2
for i in range(len(features)-20):
    if features['Mid_price'][i+20] > features['Mid_price'][i]: 
        features['Mid_price_movement'][i] = 0
    elif features['Mid_price'][i+20] == features['Mid_price'][i]: 
        features['Mid_price_movement'][i] = 1
    else:
        features['Mid_price_movement'][i] = 2

# bid-ask spread-crossing


# %%time
features['Spread_crossing'] = np.zeros((len(features), 1))
# set upward as 0, stationary as 1, downward as 2
for i in range(len(features)-20):
    if features['BidPrice1'][i+20] > features['AskPrice1'][i]: 
        features['Spread_crossing'][i] = 0
    elif features['BidPrice1'][i+20] == features['AskPrice1'][i]: 
        features['Spread_crossing'][i] = 1
    else:
        features['Spread_crossing'][i] = 2

# Delete verbose info(mid-price)
features = features.drop(['Mid_price'], axis=1)

# ## 3 Exploratory Data Analysis

# +
fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(10,10))

ax1.plot((pd.Series(features['BidPrice1'])), lw=1, color='red')
ax1.set_title ("BidPrice1 Movement", fontsize=12);
#ax1.set_xlabel ("Time(500ms)", fontsize=18)
ax1.set_ylabel ("BidPrice1", fontsize=15);

ax2.plot((pd.Series(features['AskPrice1'])), lw=1, color='blue')
ax2.set_title ("AskPrice1 Movement", fontsize=12);
#ax2.set_xlabel ("Time(500ms)", fontsize=18)
ax2.set_ylabel ("AskPrice1", fontsize=15);

ax3.plot((pd.Series(features['BidPrice1']).iloc[0:21]), lw=1, color='red',label="BidPrice1",linestyle='--')
ax3.plot((pd.Series(features['AskPrice1']).iloc[0:21]), lw=1, color='blue',label='AskPrice1',linestyle='--')
ax3.scatter(20,features['BidPrice1'][20], color='red')
ax3.scatter(0,features['AskPrice1'][0], color='blue')
ax3.set_title ("Bid-Ask Spread(0-20tick)", fontsize=12);
ax3.set_xlabel ("Time(500ms)", fontsize=10)
ax3.set_ylabel ("Price1", fontsize=15);
ax3.legend()
gc.collect();
# -

features = features.to_csv('features.csv')
