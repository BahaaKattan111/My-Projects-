from sklearn import tree, ensemble
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression, Ridge
import torch
from torch import nn
import torch.nn.functional as F
import lightgbm

lgb_params = {
    # 'metric' : 'rmse',
    'learning_rate': 0.1,
    'max_depth': 25,
    'num_leaves': 1000,
    'objective': 'regression',
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    'max_bin': 1000}
models = { 'Ridge': Ridge(alpha=0.0002), 'lightgbm': LGBMRegressor(**lgb_params), 'svr': SVR(), 'rf': ensemble.RandomForestRegressor()}
