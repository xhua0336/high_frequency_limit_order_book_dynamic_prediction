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
import seaborn as sns
import matplotlib.pyplot as plt



# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Model
from sklearn import svm
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

# Tools and metrics
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error
from sklearn.preprocessing import PowerTransformer
from skopt import BayesSearchCV

# +
#Purged Group Time Series
# TODO: make GitHub GIST
# TODO: add as dataset
# TODO: add logging with verbose

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]

# +
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
    
# this is code slightly modified from the sklearn docs here:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    
    cmap_cv = plt.cm.coolwarm

    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)   # inplace
    cmap_data = ListedColormap(jet(seq))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.Set3)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['target', 'day']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


# -

data = pd.read_csv("features.csv")
data = data.iloc[:27400,1:]

# +
# Models
from sklearn import svm
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn import set_config
set_config(display='diagram') 
# -

data

# ## Set up the Pipeline

len(data)/20

# +
metrics_name = ['accuracy_', 'f1_score_', 'average_precision_score_']
name = ["md_upward", "md_stationary", "md_downward", "sc_upward", "sc_stationary", "sc_downward"]
metrics = []

for i in range(len(metrics_name)):
    for j in range(len(name)):
        metrics.append(metrics_name[i]+name[j])
        
for i in range(len(metrics)): 
    vars()[metrics[i]]= np.zeros(100)
# -

data

# %%time
for i in range(100):  #1370-100
    X = data.iloc[i*20:i*20+2020, 0:64]
    y = data.iloc[i*20:i*20+2020,64:] 
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=20, shuffle = False)
    y_train_md = y_train.iloc[:,0]
    y_train_sc = y_train.iloc[:,1]
    y_test_md = y_test.iloc[:,0]
    y_test_sc = y_test.iloc[:,1]
    
    y_train_md = preprocessing.label_binarize(y_train_md, classes=[0, 1, 2])
    y_train_sc = preprocessing.label_binarize(y_train_sc, classes=[0, 1, 2])
    y_test_md = preprocessing.label_binarize(y_test_md, classes=[0, 1, 2])
    y_test_sc = preprocessing.label_binarize(y_test_sc, classes=[0, 1, 2])    

    y_train_md = pd.DataFrame(y_train_md)
    y_train_sc = pd.DataFrame(y_train_sc)
    y_test_md = pd.DataFrame(y_test_md)
    y_test_sc = pd.DataFrame(y_test_sc)

    y_train_md_upward = y_train_md.iloc[:,0]
    y_train_md_stationary = y_train_md.iloc[:,1]
    y_train_md_downward = y_train_md.iloc[:,2]
    
    y_train_sc_upward = y_train_sc.iloc[:,0]
    y_train_sc_stationary = y_train_sc.iloc[:,1]
    y_train_sc_downward = y_train_sc.iloc[:,2]
    
    y_test_md_upward = y_test_md.iloc[:,0]
    y_test_md_stationary = y_test_md.iloc[:,1]
    y_test_md_downward = y_test_md.iloc[:,2]
    
    y_test_sc_upward = y_test_sc.iloc[:,0]
    y_test_sc_stationary = y_test_sc.iloc[:,1]
    y_test_sc_downward = y_test_sc.iloc[:,2]

   
    scaler = StandardScaler()
    
    clf_md_upward = OneVsRestClassifier(LinearSVC())#svm.SVC()
    clf_md_starionary = OneVsRestClassifier(LinearSVC())#svm.SVC()
    clf_md_downward = OneVsRestClassifier(LinearSVC())#svm.SVC()
    clf_sc_upward = OneVsRestClassifier(LinearSVC())#svm.SVC()
    clf_sc_starionary = OneVsRestClassifier(LinearSVC())#svm.SVC()
    clf_sc_downward = OneVsRestClassifier(LinearSVC())#svm.SVC()
   
    pipe_md_upward = Pipeline(steps=[
    ('scaler', scaler),
    ('SVM', clf_md_upward)
    ])
    pipe_md_stationary = Pipeline(steps=[
    ('scaler', scaler),
    ('SVM', clf_md_starionary)
    ])
    pipe_md_downward = Pipeline(steps=[
    ('scaler', scaler),
    ('SVM', clf_md_downward)
    ])
    pipe_sc_upward = Pipeline(steps=[
    ('scaler', scaler),
    ('SVM', clf_sc_upward)
    ])
    pipe_sc_stationary = Pipeline(steps=[
    ('scaler', scaler),
    ('SVM', clf_sc_starionary)
    ])
    pipe_sc_downward = Pipeline(steps=[
    ('scaler', scaler),
    ('SVM', clf_sc_downward)
    ])
    
    clf_SVM = OneVsRestClassifier(LinearSVC())
    params = {
        'estimator__C': [0.5, 1.0, 1.5],
        'estimator__tol': [1e-3, 1e-4, 1e-5],
        }
    gs = GridSearchCV(clf_SVM, params, cv=5) #, scoring='roc_auc'
    
    pipe_md_upward.fit(X_train,y_train_md_upward)
    pipe_md_stationary.fit(X_train,y_train_md_stationary)
    pipe_md_downward.fit(X_train,y_train_md_downward)
    pipe_sc_upward.fit(X_train,y_train_sc_upward)
    pipe_sc_stationary.fit(X_train,y_train_sc_stationary)
    pipe_sc_downward.fit(X_train,y_train_sc_downward)

    pred_md_upward = pipe_md_upward.predict(X_test.values)
    pred_md_stationary = pipe_md_stationary.predict(X_test.values)
    pred_md_downward = pipe_md_downward.predict(X_test.values)
    
    pred_sc_upward = pipe_sc_upward.predict(X_test.values)
    pred_sc_stationary = pipe_sc_stationary.predict(X_test.values)
    pred_sc_downward = pipe_sc_downward.predict(X_test.values)
    
    accuracy_md_upward[i] = accuracy_score(y_test_md_upward,pred_md_upward)
    accuracy_md_stationary[i] = accuracy_score(y_test_md_stationary,pred_md_stationary)
    accuracy_md_downward[i] = accuracy_score(y_test_md_downward,pred_md_downward)
   
    accuracy_sc_upward[i] = accuracy_score(y_test_sc_upward,pred_sc_upward)
    accuracy_sc_stationary[i] = accuracy_score(y_test_sc_stationary,pred_sc_stationary)
    accuracy_sc_downward[i] = accuracy_score(y_test_sc_downward,pred_sc_downward)
    
    f1_score_md_upward[i] = f1_score(y_test_md_upward,pred_md_upward)
    f1_score_md_stationary[i] = f1_score(y_test_md_stationary,pred_md_stationary)
    f1_score_md_downward[i] = f1_score(y_test_md_downward,pred_md_downward)
   
    f1_score_sc_upward[i] = f1_score(y_test_sc_upward,pred_sc_upward)
    f1_score_sc_stationary[i] = f1_score(y_test_sc_stationary,pred_sc_stationary)
    f1_score_sc_downward[i] = f1_score(y_test_sc_downward,pred_sc_downward)
    
    average_precision_score_md_upward[i] = precision_score(y_test_md_upward,pred_md_upward)
    average_precision_score_md_stationary[i] = precision_score(y_test_md_stationary,pred_md_stationary)
    average_precision_score_md_downward[i] = precision_score(y_test_md_downward,pred_md_downward)
   
    average_precision_score_sc_upward[i] = precision_score(y_test_sc_upward,pred_sc_upward)
    average_precision_score_sc_stationary[i] = precision_score(y_test_sc_stationary,pred_sc_stationary)
    average_precision_score_sc_downward[i] = precision_score(y_test_sc_downward,pred_sc_downward)

# +
mean_accuracy_md_upward = np.mean(accuracy_md_upward)
mean_accuracy_md_stationary = np.mean(accuracy_md_stationary)
mean_accuracy_md_downward = np.mean(accuracy_md_downward)

mean_accuracy_sc_upward = np.mean(accuracy_sc_upward)
mean_accuracy_sc_stationary = np.mean(accuracy_sc_stationary)
mean_accuracy_sc_downward = np.mean(accuracy_sc_downward)

mean_f1_score_md_upward = np.mean(f1_score_md_upward)
mean_f1_score_md_stationary = np.mean(f1_score_md_stationary)
mean_f1_score_md_downward = np.mean(f1_score_md_downward)

mean_f1_score_sc_upward = np.mean(f1_score_sc_upward)
mean_f1_score_sc_stationary = np.mean(f1_score_sc_stationary)
mean_f1_score_sc_downward = np.mean(f1_score_sc_downward)

mean_precision_score_md_upward = np.mean(average_precision_score_md_upward)
mean_precision_score_md_stationary = np.mean(average_precision_score_md_stationary)
mean_precision_score_md_downward = np.mean(average_precision_score_md_downward)

mean_precision_score_sc_upward = np.mean(average_precision_score_sc_upward)
mean_precision_score_sc_stationary = np.mean(average_precision_score_sc_stationary)
mean_precision_score_sc_downward = np.mean(average_precision_score_sc_downward)
    
    
# Initialise table
columns=['Upward_Accuracy', 'Upward_Precision','Upward_f1_score']
rows=['Mid_price_movement', 'Bid_ask_spread']
results =pd.DataFrame(0.0, columns=columns, index=rows)

# Computer metrics
results.iloc[0, 0] = mean_accuracy_md_upward
results.iloc[0, 1] = mean_precision_score_md_upward
results.iloc[0, 2] = mean_f1_score_md_upward
results.iloc[1, 0] = mean_accuracy_sc_upward
results.iloc[1, 1] = mean_precision_score_sc_upward
results.iloc[1, 2] = mean_f1_score_sc_upward
    
results.round(4)

# +
# Initialise table
columns=['Stationary_Accuracy', 'Stationary_Precision','Stationary_f1_score']
rows=['Mid_price_movement', 'Bid_ask_spread']
results =pd.DataFrame(0.0, columns=columns, index=rows)

# Computer metrics
results.iloc[0, 0] = mean_accuracy_md_stationary
results.iloc[0, 1] = mean_precision_score_md_stationary
results.iloc[0, 2] = mean_f1_score_md_stationary
results.iloc[1, 0] = mean_accuracy_sc_stationary
results.iloc[1, 1] = mean_precision_score_sc_stationary
results.iloc[1, 2] = mean_f1_score_sc_stationary
    
results.round(4)

# +
# Initialise table
columns=['Downard_Accuracy', 'Downard_Precision','Downard_f1_score']
rows=['Mid_price_movement', 'Bid_ask_spread']
results =pd.DataFrame(0.0, columns=columns, index=rows)

# Computer metrics
results.iloc[0, 0] = mean_accuracy_md_downward
results.iloc[0, 1] = mean_precision_score_md_downward
results.iloc[0, 2] = mean_f1_score_md_downward
results.iloc[1, 0] = mean_accuracy_sc_downward
results.iloc[1, 1] = mean_precision_score_sc_downward
results.iloc[1, 2] = mean_f1_score_sc_downward
    
results.round(4)
# -

# ## 模型回测

open_price = data['']
open_price_test = open_price.iloc[59767:74584].reset_index(drop=True)
#open_price_test

from collections import OrderedDict
def get_daily_pnl(testx,testy, period=5, tranct_ratio=False,threshold=0.001, tranct=1.1e-4, noise=0, notional=False,invest = 100):
    n_bar = len(testx)
    price = open_price_test#pd.Series(testx['ClosePrice'].astype('int64')).reset_index(drop=True)
    
    #过去5分钟收益率（滚动）
    ret_5 = (testy.rolling(period).sum()).dropna().reset_index(drop=True)
    ret_5 = ret_5.append(pd.Series([0]*(len(testy)-len(ret_5)))).reset_index(drop=True) 
    #ret_5 = testy
    
    #交易信号 过去5分钟收益大于阈值买入，过去5分钟收益小于负阈值卖出
    signal = pd.Series([0] * n_bar)
    signal[(ret_5>threshold)] = 1
    signal[(ret_5< -threshold)] = -1
   
    #买仓
    position_pos = pd.Series([np.nan] * n_bar)
    position_pos[0] = 0 
    position_pos[(signal==1)] = 1
    position_pos[(ret_5< -threshold)] = 0
    position_pos.ffill(inplace=True)
    
    pre_pos = position_pos.shift(1)#前一分钟持仓情况
    position_pos[(position_pos==1) & (pre_pos==1)] = np.nan #如果前一分钟持有，并且交易信号是1，不执行交易
    position_pos[(position_pos==1)] = invest/price[(position_pos==1)]
    position_pos.ffill(inplace=True)
        
    #卖仓
    position_neg = pd.Series([np.nan] * n_bar)
    position_neg[0] = 0
    position_neg[(signal==-1)] = -1
    position_neg[(ret_5> threshold)] = 0
    position_neg.ffill(inplace=True)
    
    pre_neg = position_neg.shift(1)
    position_neg[(position_neg==-1) & (pre_neg==-1)] = np.nan
    position_neg[(position_neg==-1)] = -invest/price[(position_neg==-1)]
    position_neg.ffill(inplace=True)
    
    #持仓
    position = position_pos + position_neg
    position[0]=0
    position[n_bar-1] = 0 #交易结束前平仓
    position[n_bar-2] = 0
    change_pos = position - position.shift(1)
    change_pos[0] = 0
    change_base = pd.Series([0] * n_bar)
    change_buy = change_pos>0
    change_sell = change_pos<0

    if (tranct_ratio):
        change_base[change_buy] = price[change_buy]*(1+tranct)
        change_base[change_sell] = price[change_sell]*(1-tranct)
    else:
        change_base[change_buy] = price[change_buy]+tranct
        change_base[change_sell] = price[change_sell]-tranct
    
    final_pnl = -sum(change_base*change_pos)
    pln_invest = final_pnl/invest
    turnover = sum(change_base*abs(change_pos))
    num = sum((position!=0) & (change_pos!=0))
    hld_period = sum(position!=0)
  
    ## finally we combine the statistics into a data frame
    #result = pd.DataFrame({"final.pnl": final_pnl, "turnover": turnover, "num": num, "hld.period": hld_period}, index=[0])
    #result = {"date": date, "final.pnl": final_pnl, "turnover": turnover, "num": num, "hld.period": hld_period}
    result = OrderedDict([ ("pln/invest", pln_invest),("final.pnl", final_pnl), ("turnover", turnover), ("num", num), ("hld.period", hld_period)])
    return result


# ###### 

# +
# Initialise table
columns=['pln/invest', 'final.pnl', 'turnover','num','hld.period']
rows=['OLS', 'Random Forest', 'XGBoost(Bayes)', 'XGBoost']
results =pd.DataFrame(0.0, columns=columns, index=rows)

# List algorithms
preds = [predict_ols, predict_rf, predict_xgbt, predict_xg] 
# Compute test predictions and metrics
for i in range(len(preds)):
    results.loc[rows[i]] = pd.DataFrame(get_daily_pnl(x_test,testy = preds[i], period=5, tranct_ratio=True, 
                                               threshold= 0.0005, tranct=0.00015, notional=True, invest = 100),index=[rows[i]]).iloc[0,:]
    
results
# -









import xgboost as xgb
import optuna
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

cv = PurgedGroupTimeSeriesSplit(
    n_splits=5,
    max_train_group_size=1250,
    group_gap=0,
    max_test_group_size=150
)

# +
fig, ax = plt.subplots()

cv = PurgedGroupTimeSeriesSplit(
    n_splits=5,
    max_train_group_size=1250,
    group_gap=0,
    max_test_group_size=150
)

plot_cv_indices(
    cv,
    X_train,y_train_md,
    X_train.index.values,
    ax,
    5,
    lw=20
);

# +
##Fit the XGBoost Classifier with Optimal Hyperparams
scaler = StandardScaler()

clf = xgb.XGBRegressor(**best_params)

pipe_xgb = Pipeline(steps=[
    ('scaler', scaler),
    ('xgb', clf)
])

pipe_xgb.fit(x_train,y_train)

gc.collect()
# -

#     clf_SVM = OneVsRestClassifier(LinearSVC())
#     params = {
#         'estimator__C': [0.5, 1.0, 1.5],
#         'estimator__tol': [1e-3, 1e-4, 1e-5],
#         }
#     gs = GridSearchCV(clf_SVM, params, cv=5) #, scoring='roc_auc'  
#
#         
#     
#     groups = X_train.index.values
#     for i, (train_idx, valid_idx) in enumerate(cv.split(
#         X_train,
#         y_labels,
#         groups=groups)):
#         
#         train_data = X_train[train_idx, :], y_labels[train_idx]
#         valid_data = X_train[valid_idx, :], y_labels[valid_idx]
#         
#         _ = pipe.fit(X_train[train_idx, :], y_labels[train_idx])
#         preds = pipe.predict(X_train[valid_idx, :])
