from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, \
    StackingRegressor, VotingRegressor
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import pandas as pd
import seaborn as sb
import numpy as pp

sb.set()
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 200)

train = pd.read_csv('train_clean_2.csv')
sub = pd.read_csv('sample_submission.csv')
test = pd.read_csv('test_clean_2.csv')

# -------- Preprocessing Missing Values ------- #

train['df'] = 'train'
test['df'] = 'test'
df = pd.concat([train, test], axis=0)

cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'BsmtFinType1',
        'GarageQual', 'GarageCond', 'GarageQual', 'PoolQC', 'Fence', 'MiscFeature']

for col in df[cols]:
    df[col].fillna('None', axis=0, inplace=True)

# ------------- #
# col: MasVnrType
df['MasVnrArea'].fillna(0, axis=0, inplace=True)

# ------------- #
# col: GarageYrBlt
df['GarageYrBlt'].fillna(df['YearBuilt'], axis=0, inplace=True)

# -------------- #
# col: Utilities
df['Utilities'].fillna(df['Utilities'].mode, axis=0, inplace=True)

# -------------- #
# col: MSZoning
df['MSZoning'].fillna('C (all)', axis=0, inplace=True)

# -------------- #
# col: KitchenQual
df['KitchenQual'].fillna('TA', axis=0, inplace=True)

# -------------- #
# col: SaleType
df['SaleType'].fillna('Oth', axis=0, inplace=True)

# -------------- #
# col: Functional
df['Functional'].fillna('Typ', axis=0, inplace=True)

# -------------- #
# col: Exterior1st
df['Exterior1st'].fillna('VinylSd', axis=0, inplace=True)

# -------------- #
# col: Exterior2nd
df['Exterior2nd'].fillna('VinylSd', axis=0, inplace=True)

# -------------- #
# col: Electrical
df['Electrical'].fillna('SBrkr', axis=0, inplace=True)

# ------------- #
cat = df.select_dtypes('object')
num_df = df.drop(cat, axis=1)

imputer = KNNImputer(n_neighbors=5)

imputing = imputer.fit_transform(num_df)
df_clean = pd.DataFrame(imputing, columns=num_df.columns)

for col in df.drop('SalePrice', axis=1):
    if df[col].dtypes != object:
        df[col] = df_clean[col]

# -------- Exploratory Data Analysis -------- #

for col in df:
    if df[col].dtypes == object:
        sb.countplot(df[col], hue=df['df'])
    else:
        try:
            plt.subplot(211)
            sb.distplot(df.loc[df['df'] == 'test', col], color='red', label='test')
            plt.legend()

            plt.subplot(212)
            sb.distplot(df.loc[df['df'] == 'train', col], color='blue', label='train')
            plt.legend()
        except:
            plt.title(df[col])
            plt.subplot(211)
            sb.kdeplot(df.loc[df['df'] == 'test', col], color='red', label='test', bw=0.3,)
            plt.legend()

            plt.subplot(212)
            sb.kdeplot( df.loc[df['df'] == 'train', col], color='blue', label='train', bw=0.3)
            plt.legend()
    plt.show()


# look for effect of every value in each feature on the price - reduce the imbalance features
plt.figure(figsize=(15, 20))
corr = df.corr().sort_values('SalePrice', ascending=False)
sb.heatmap(corr)
plt.show()


# --------- Detect Outliers  ---------- #
def detect_outliers(df, target):
    for col in df:

        plt.figure(figsize=(14, 11))
        plt.title(col)
        plt.suptitle('Detect Outliers')

        if df[col].dtypes == object:
            sb.boxplot(df[col], df[target], color='gray')
            sb.boxenplot(df[col], df[target])
        else:
            plt.subplot(311)
            try:
                sb.distplot(df[col], hist=False, rug=True)
            except:

                sb.kdeplot(df[col], bw=0.3)

            plt.subplot(312)
            sb.scatterplot(df[col], df[target])

            plt.subplot(313)
            sb.boxenplot(df[col])
            sb.boxplot(df[col], color='gray')

        plt.show()

# --------- Remove Outliers  ---------- #

df.loc[(df['MSZoning'] == 'RL') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan
df.loc[(df['MSZoning'] == 'RM') & (df['SalePrice'] > 350000), 'SalePrice'] = pp.nan
df.loc[(df['MSZoning'] == 'RM') & (df['SalePrice'] > 350000), 'SalePrice'] = pp.nan

# ---------------- #
# col: LotShape
df.loc[(df['LotShape'] == 'Reg') & (df['SalePrice'] > 500000), 'SalePrice'] = pp.nan
df.loc[(df['LotShape'] == 'LR2') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['LotShape'] == 'IR1') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan

# ---------------- #
# col: LandContour
df.loc[(df['LotArea'] == 'Lvl') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan
df.loc[(df['LotArea'] == 'Bnk') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['LotArea'] == 'HLS') & (df['SalePrice'] > 500000), 'SalePrice'] = pp.nan

# ---------------- #
# col: Utilities
df.drop('Utilities', axis=1, inplace=True)
# ---------------- #
# col: LotConfig
df.loc[(df['LotConfig'] == 'FR2') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['LotConfig'] == 'Corner') & (df['SalePrice'] > 350000), 'SalePrice'] = pp.nan
df.loc[(df['LotConfig'] == 'CulDSac') & (df['SalePrice'] > 350000), 'SalePrice'] = pp.nan

# ---------------- #
# col: LandSlope
df.drop('LandSlope', axis=1, inplace=True)

# --------------- #
# col: Neighborhood
df.loc[(df['Neighborhood'] == 'CollgCr') & (df['SalePrice'] > 350000), 'SalePrice'] = pp.nan
df.loc[(df['Neighborhood'] == 'Crawfor') & (df['SalePrice'] > 350000), 'SalePrice'] = pp.nan
df.loc[(df['Neighborhood'] == 'NoRidge') & (df['SalePrice'] > 500000), 'SalePrice'] = pp.nan
df.loc[(df['Neighborhood'] == 'NAmes') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['Neighborhood'] == 'Oldtown') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['Neighborhood'] == 'Edwards') & (df['SalePrice'] > 250000), 'SalePrice'] = pp.nan
df.loc[(df['Neighborhood'] == 'Gilbert') & (df['SalePrice'] > 250000), 'SalePrice'] = pp.nan

# ------------- #
# col: Condition1
df.loc[(df['Condition1'] == 'Norm') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan
df.loc[(df['Condition1'] == 'PosN') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['Condition1'] == 'Artery') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['Condition1'] == 'RRAn') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['Condition1'] == 'PosA') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan

# ------------- #
# col: Condition2
df.loc[(df['Condition2'] == 'Norm') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan

# ------------- #
# col: BldgType
df.loc[(df['BldgType'] == '1Fam') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan
df.loc[(df['BldgType'] == 'TwnhsE') & (df['SalePrice'] > 350000), 'SalePrice'] = pp.nan

# ------------- #
# col: HouseStyle
df.loc[(df['BldgType'] == '2Story') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan
df.loc[(df['BldgType'] == '1Story') & (df['SalePrice'] > 500000), 'SalePrice'] = pp.nan
df.loc[(df['BldgType'] == '1.5Story') & (df['SalePrice'] > 350000), 'SalePrice'] = pp.nan
df.loc[(df['BldgType'] == 'SLvl') & (df['SalePrice'] > 250000), 'SalePrice'] = pp.nan
df.loc[(df['BldgType'] == '2.5Unf') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['BldgType'] == '2.5Fin') & (df['SalePrice'] > 450000), 'SalePrice'] = pp.nan
df.loc[(df['BldgType'] == '1.5Fin') & (df['SalePrice'] > 340000), 'SalePrice'] = pp.nan

# ------------- #
# col: RoofStyle
df.loc[(df['RoofStyle'] == 'Gable') & (df['SalePrice'] > 500000), 'SalePrice'] = pp.nan
df.loc[(df['RoofStyle'] == 'Hip') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan
df.loc[(df['RoofStyle'] == 'Hip') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan

# ------------- #
# col: RoofMatl
df.loc[(df['RoofStyle'] == 'Hip') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan
df.loc[(df['RoofStyle'] == 'WdShngl') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan

# ------------- #
# col: Exterior1st
df.loc[(df['Exterior1st'] == 'VinylSd') & (df['SalePrice'] > 500000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior1st'] == 'Wd Sdng') & (df['SalePrice'] > 400000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior1st'] == 'HdBoard') & (df['SalePrice'] > 400000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior1st'] == 'BrkFace') & (df['SalePrice'] > 400000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior1st'] == 'WdShing') & (df['SalePrice'] > 250000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior1st'] == 'CemntBd') & (df['SalePrice'] > 400000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior1st'] == 'Plywood') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior1st'] == 'stucco') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan

# --------------- #
# col: Exterior2st
df.loc[(df['Exterior2nd'] == 'VinylSd') & (df['SalePrice'] > 500000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior2nd'] == 'Wd Sdng') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior2nd'] == 'HdBoard') & (df['SalePrice'] > 400000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior2nd'] == 'BrkFace') & (df['SalePrice'] > 400000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior2nd'] == 'WdShing') & (df['SalePrice'] > 350000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior2nd'] == 'CemntBd') & (df['SalePrice'] > 400000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior2nd'] == 'Plywood') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior2nd'] == 'Stucco') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior2nd'] == 'AsbShng') & (df['SalePrice'] > 200000), 'SalePrice'] = pp.nan
df.loc[(df['Exterior2nd'] == 'ImStucc') & (df['SalePrice'] > 400000), 'SalePrice'] = pp.nan

# -------------- #
# col: MasVnrType
df.loc[(df['MasVnrType'] == 'BrkFace') & (df['SalePrice'] > 500000), 'SalePrice'] = pp.nan
df.loc[(df['MasVnrType'] == 'None') & (df['SalePrice'] > 500000), 'SalePrice'] = pp.nan
df.loc[(df['MasVnrType'] == 'Stone') & (df['SalePrice'] > 500000), 'SalePrice'] = pp.nan
df.loc[(df['MasVnrType'] == 'BrkCmn') & (df['SalePrice'] > 230000), 'SalePrice'] = pp.nan

# -------------- #
# col: Foundation
df.loc[(df['Foundation'] == 'PConc') & (df['SalePrice'] > 610000), 'SalePrice'] = pp.nan
df.loc[(df['Foundation'] == 'BrkTil') & (df['SalePrice'] > 400000), 'SalePrice'] = pp.nan

# -------------- #
# col: BsmtFinType1
df.loc[(df['BsmtFinType1'] == 'GLQ') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan
df.loc[(df['BsmtFinType1'] == 'ALQ') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['BsmtFinType1'] == 'Unf') & (df['SalePrice'] > 500000), 'SalePrice'] = pp.nan
df.loc[(df['BsmtFinType1'] == 'Rec') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['BsmtFinType1'] == 'BLQ') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan

# -------------- #
# col: BsmtFinType2
df.loc[(df['BsmtFinType2'] == 'LwQ') & (df['SalePrice'] > 2800000), 'SalePrice'] = pp.nan
df.loc[(df['BsmtFinType2'] == 'ALQ') & (df['SalePrice'] > 320000), 'SalePrice'] = pp.nan
df.loc[(df['BsmtFinType2'] == 'Unf') & (df['SalePrice'] > 600000), 'SalePrice'] = pp.nan
df.loc[(df['BsmtFinType2'] == 'Rec') & (df['SalePrice'] > 300000), 'SalePrice'] = pp.nan
df.loc[(df['BsmtFinType2'] == 'BLQ') & (df['SalePrice'] > 2300000), 'SalePrice'] = pp.nan

# ------------ #
# col: GarageYrBlt
df.loc[df['GarageYrBlt'] >= 2100, 'GarageYrBlt'] = 2007

# ------------ #
# col: WoodDeckSF
df.loc[df['WoodDeckSF'] >= 1000, 'WoodDeckSF'] = pp.nan

# ------------ #
# col: Fence
df.loc[(df['Fence'] == 'GdWo') & (df['SalePrice'] > 200000), 'SalePrice'] = pp.nan

# ---------- Preprocessing: converting text data to numerical data --------- #

# --------Preprocessing Ordinal Data -------- #

# col: MSZoning
df['MSZoning'] = df['MSZoning'].map({'RL': 5, 'RM': 3, 'C (all)': 1, 'FV': 4, 'RH': 2})
# ------------- #

# col: 'Street', 'Alley'
for col in df[['Street', 'Alley']]:
    df[col] = df[col].map({'Pave': 2, 'Grvl': 1, 'None': 0})
# ------------ #

# col: LotShape
df['LotShape'] = df['LotShape'].map({'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1, })
# ----------- #

# col: LandContour
df['LandContour'] = df['LandContour'].map({'Lvl': 1, 'Bnk': 2, 'Low': 3, 'HLS': 4})
# ------------ #

# col: LotConfig
df['LotConfig'] = df['LotConfig'].map({'Inside': 1, 'FR2': 4, 'Corner': 2, 'CulDSac': 3, 'FR3': 5})
# ----------- #

# col: ['PoolQC', 'GarageCond', 'GarageQual', 'ExterQual', 'ExterCond',
#                'BsmtQual', 'FireplaceQu', 'BsmtCond', 'HeatingQC', 'KitchenQual']
for col in df[['PoolQC', 'GarageCond', 'GarageQual', 'ExterQual', 'ExterCond',
               'BsmtQual', 'FireplaceQu', 'BsmtCond', 'HeatingQC', 'KitchenQual']]:
    df[col] = df[col].map({'Gd': 4, 'TA': 3, 'Ex': 5, 'Fa': 2, 'Po': 1, 'None': 0})
# ----------- #

# col: BsmtExposure
df['BsmtExposure'] = df['BsmtExposure'].map({'Gd': 4, 'Mn': 2, 'Av': 3, 'None': 0, 'No': 1})
# ----------- #

# col: 'BsmtFinType1', 'BsmtFinType2'
for col in df[['BsmtFinType1', 'BsmtFinType2']]:
    df[col] = df[col].map({'GLQ': 6, 'ALQ': 5, 'Unf': 1, 'Rec': 3, 'BLQ': 4, 'None': 0, 'LwQ': 2})
# ----------- # 

# col: Heating
df['Heating'] = df['Heating'].map({'GasA': 4, 'GasW': 3, 'Grav': 2, 'Wall': 1, 'OthW': 1, 'Floor': 1})
# ----------- #

# col: CentralAir
df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N': 0})
# ----------- #

# col: Electrical
df['Electrical'] = df['Electrical'].map({'SBrkr': 5, 'FuseF': 3, 'FuseA': 4, 'FuseP': 2, 'Mix': 1})
# ----------- #

# col: GarageFinish
df['GarageFinish'] = df['GarageFinish'].map({'RFn': 2, 'Unf': 1, 'Fin': 3, 'None': 0})
# ----------- #

# col: PavedDrive
df['PavedDrive'] = df['PavedDrive'].map({'Y': 2, 'P': 1, 'N': 0})

# col: Fence
df['Fence'] = df['Fence'].map({'None': 0, 'MnPrv': 3, 'GdWo': 2, 'GdPrv': 4, 'MnWw': 1})
# ----------- #

# -------- Preprocessing Categorical Data -------- #



for col in df.select_dtypes(object):
    dummy = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummy], axis=1)
    df.drop(col, axis=1, inplace=True)

for col in df.select_dtypes(object):
    label = LabelEncoder()
    df[col] = label.fit_transform(df[col])
print(df.head())
# -------- Fix the Removed Outliers by KNNImputer -------- #

imputer = KNNImputer(n_neighbors=9)

imputing = imputer.fit_transform(pp.array(df['SalePrice']).reshape(-1, 1))
df['SalePrice'] = imputing

# save the clean data to new file
train = df[df['df'] == 1]
test = df[df['df'] == 0]

train.drop('df', axis=1, inplace=True)
test.drop('df', axis=1, inplace=True)
print(train)
# train.to_csv('train_clean_2.csv', index=False)
# test.to_csv('test_clean_2.csv', index=False)



# ---------- Feature Inginereeing -------- #

df = train

df['HasShed'] = df.MiscFeature.apply(lambda x: 1 if x == 'Shed' else 0)

df['HasTennis'] = df.MiscFeature.apply(lambda x: 1 if x == 'TenC' else 0)

df['HasGar2'] = df.MiscFeature.apply(lambda x: 1 if x == 'Gar2' else 0)

df['HasPool'] = df.PoolArea.apply(lambda x: 1 if x > 0 else 0)

df['HasDeck'] = df.WoodDeckSF.apply(lambda x: 1 if x > 0 else 0)

df['IsNew'] = df.YearBuilt.apply(lambda x: 1 if x > 2000 else 0)

df['IsOld'] = df.YearBuilt.apply(lambda x: 1 if x < 1946 else 0)

df.drop('MiscFeature', axis=1, inplace=True)

# ------------------------------- #

df['Age'] = df['YrSold'] - df['YearBuilt']

df['BsmtTotalBathRooms'] = df['BsmtFullBath'] + df['BsmtHalfBath']

df['AbvGradeTotalBathRooms'] = df['FullBath'] + df['HalfBath']

df['Total Rooms'] = df['BedroomAbvGr'] + df['BsmtFullBath'] + df['BsmtHalfBath'] + \
                    df['FullBath'] + df['HalfBath'] + df['TotRmsAbvGrd'] + \
                    df['KitchenAbvGr']

# -------------------- Selecting Best Models -------------- #
scale = StandardScaler()
X = scale.fit_transform(df.drop(['SalePrice', 'Id'], axis=1))
y = train.SalePrice

model = RandomForestRegressor()
scores = cross_val_score(model, X, y, cv=5)
print('RandomForestRegressor mean:  ', pp.mean(scores), ' STD: ', pp.std(scores))

# ---------- #
model = ExtraTreesRegressor()
scores = cross_val_score(model, X, y, cv=5)
print('ExtraTreesRegressor mean:  ', pp.mean(scores), ' STD: ', pp.std(scores))

# ---------- #
model = GradientBoostingRegressor()
scores = cross_val_score(model, X, y, cv=5)
print('GradientBoostingRegressor mean:  ', pp.mean(scores), ' STD: ', pp.std(scores))

# ---------- #
model = XGBRegressor()
scores = cross_val_score(model, X, y, cv=5)
print('XGBRegressor mean:  ', pp.mean(scores), ' STD: ', pp.std(scores))

# ---------- #
model = LGBMRegressor()
scores = cross_val_score(model, X, y, cv=5)
print('LGBMRegressor mean:  ', pp.mean(scores), ' STD: ', pp.std(scores))

# ------------------- Tuning Hyper-parameters ------------- #

RFdict = {'bootstrap': [True, False],
          'max_depth': pp.arange(55, 79),
          'max_features': [0.4, 0.2, 0.3, 0.1],
          'min_samples_leaf': [2, 3, 1],
          'min_samples_split': [0.3, 0.2, 0.1],
          'n_estimators': pp.arange(140, 160)}

ExtraRFdict = {'bootstrap': [True, False],
               'max_depth': pp.arange(80, 120),
               'max_features': ['sqrt', 0.3, 0.8, 0.5],
               'min_samples_leaf': [2, 3, 1],
               'min_samples_split': [.3, 0.5, 1., 0.8],
               'n_estimators': pp.arange(180, 220)}

GBdict = {'learning_rate': [1.0, 0.4, 0.1, 0.01, 0.03, 0.001],
          'max_depth': pp.arange(35, 50),
          'max_features': ['auto', 'sqrt', 0.5, 0.3, 0.8],
          'min_samples_leaf': [8, 6, 4],
          'min_samples_split': [.3, 0.5, 1., 0.8],
          'n_estimators': pp.arange(400, 500, 10)}

LGBMdict = {'num_leaves': pp.arange(100, 115),
            'min_data_in_leaf': pp.arange(1, 4),
            'max_depth': pp.arange(25, 40),
            'feature_fraction': [0.6],
            'max_bin': pp.arange(8, 15),
            'learning_rate ': [0.3, 0.2, 0.1]}

XGboostdict = {'nthread': [4],  # when use hyperthread, xgboost may become slower
               'learning_rate': [.03, 0.04, 0.035],  # so called `eta` value
               'max_depth': pp.arange(25, 33),
               'min_child_weight': [9, 8, 7],
               'silent': [1],
               'subsample': [0.55, 0.5, 0.45],
               'n_estimators': pp.arange(145, 155)}


# ---------------- GridSearchCV  -----------------#

LGBM = GridSearchCV(LGBMRegressor(), LGBMdict, cv=3, scoring='neg_mean_squared_error')
LGBM.fit(X, y)
print('LGBM: ', LGBM.best_score_, LGBM.best_params_)
print()


# XGBRegressor:
XGB = GridSearchCV(XGBRegressor(), XGboostdict, cv=3, scoring='neg_mean_squared_error')
XGB.fit(X, y)
print('XGB: ', XGB.best_score_, XGB.best_params_)
print()

# GradientBoostingRegressor:
GB = GridSearchCV(GradientBoostingRegressor(), GBdict, cv=3, scoring='neg_mean_squared_error')
GB.fit(X, y)
print('GB: ', GB.best_score_, GB.best_params_)
print()

# ------------ #

# RandomForestRegressor:
RFR = GridSearchCV(RandomForestRegressor(), RFdict,  cv=3, scoring='neg_mean_squared_error')
RFR.fit(X, y)
print('RandomForestRegressor: ', RFR.best_score_, RFR.best_params_)

print()
# ----------- #
# ExtraTreesRegressor:
ExtraRFR = GridSearchCV(ExtraTreesRegressor(), ExtraRFdict, cv=3, scoring='neg_mean_squared_error')
ExtraRFR.fit(X, y)
print('ExtraRFR: ', ExtraRFR.best_score_, ExtraRFR.best_params_)

# ----------------------- cross_val_score ----------------- #

model =  LGBM.best_estimator_
cores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print('Ridge mean:  ', pp.mean(scores), ' STD: ', pp.std(scores))

model = GB.best_estimator_
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print('GradientBoostingRegressor mean:  ', pp.mean(scores), ' STD: ', pp.std(scores))

model = XGB.best_estimator_
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print('XGBRegressor mean:  ', pp.mean(scores), ' STD: ', pp.std(scores))

model = RFR.best_estimator_
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print('RandomForestRegressor mean:  ', pp.mean(scores), ' STD: ', pp.std(scores))

model = ExtraRFR.best_estimator_
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print('ExtraTreesRegressor mean:  ', pp.mean(scores), ' STD: ', pp.std(scores))


# -------------- Selecting Best Features that affect the accuracy of our models -------------- #

model = ExtraRFR.best_estimator_
model2 = RFR.best_estimator_
model3 = GB.best_estimator_
model4 = XGB.best_estimator_
model5 = LGBM.best_estimator_


model.fit(X, y)
model2.fit(X, y)
model3.fit(X, y)
model4.fit(X, y)
model5.fit(X, y)

# -------- Feature Importances --------- #
Importance1 = pd.Series(model.feature_importances_, index=df.drop(['SalePrice', 'Id'], axis=1).columns)
Importance2 = pd.Series(model2.feature_importances_, index=df.drop(['SalePrice', 'Id'], axis=1).columns)
Importance3 = pd.Series(model3.feature_importances_, index=df.drop(['SalePrice', 'Id'], axis=1).columns)
Importance4 = pd.Series(model4.feature_importances_, index=df.drop(['SalePrice', 'Id'], axis=1).columns)
Importance5 = pd.Series(model5.feature_importances_, index=df.drop(['SalePrice', 'Id'], axis=1).columns)
Importances = pd.Series(Importance1 + Importance2 + Importance3 + Importance4 + Importance5)


# ------ Extract Best Cols with the highest score and best clf ------ #
best_clf = dict(cols=[], score=[], clf=[])

cv = 15
for i in range(1, len(Importances.index)):
    X = df[Importances.nlargest(i).index]

    scores = cross_val_score(model, X, y, cv=cv)
    best_clf['clf'].append('RandomForestRegressor')
    best_clf['cols'].append(i)
    best_clf['score'].append(pp.mean(scores))

    scores = cross_val_score(model2, X, y, cv=cv)
    best_clf['clf'].append('ExtraTreesRegressor')
    best_clf['cols'].append(i)
    best_clf['score'].append(pp.mean(scores))

    scores = cross_val_score(model3, X, y, cv=cv)
    best_clf['clf'].append('GradientBoostingRegressor')
    best_clf['cols'].append(i)
    best_clf['score'].append(pp.mean(scores))

    scores = cross_val_score(model4, X, y, cv=cv)
    best_clf['clf'].append('XGBRegressor')
    best_clf['cols'].append(i)
    best_clf['score'].append(pp.mean(scores))

    scores = cross_val_score(model5, X, y, cv=cv)
    best_clf['clf'].append('ExtraRFC')
    best_clf['cols'].append(i)
    best_clf['score'].append(pp.mean(scores))

best_clf = pd.DataFrame(best_clf).sort_values('score', ascending=False).drop_duplicates('clf')
print(best_clf[:10])
# every model has different cols that affect his accuaracy so
# in the next section we will take all the models and fit them on the best cols was chosen from each model (best_clf).

# ------------------ Select best cols that affect all the models ------------------ #

ExtraTrees = ExtraRFR.best_estimator_
RandomForest = RFR.best_estimator_
GB = GB.best_estimator_
XGBoost = XGB.best_estimator_
LightGBM = LGBM.best_estimator_

colslist = []
scorelist = []

for cols in best_clf['cols']:
    stack = StackingRegressor([('ExtraTrees', ExtraTrees), ('RandomForest', RandomForest), ('GB', GB),
                               ('XGBoost', XGBoost), ('LightGBM', LightGBM)])
    X = scale.fit_transform(df[Importances.nlargest(int(cols)).index])
    scores = cross_val_score(stack, X, y, cv=cv)

    colslist.append(cols)
    scorelist.append(pp.mean(scores))

    print(f'cols: {cols}')
    print('Stacking mean:', pp.mean(scores), 'STD: ', pp.std(scores))

bestcols = pd.DataFrame({'cols': colslist, 'score': scorelist})
print(bestcols)

# ------------ Extract cols with best score and use it for last submission ------------ #
best_col = bestcols.loc[bestcols['score'] == pp.max(bestcols['score']), 'cols']
print(best_col)

# ------------ Combine Best models using 'StackingRegressor' and 'VotingRegressor' ------------ #

# ----------- StackingRegressor ----------- #

stack = StackingRegressor([('ExtraTrees', ExtraTrees), ('RandomForest', RandomForest), ('GB', GB),
                           ('XGBoost', XGBoost), ('LightGBM', LightGBM)])

X = scale.fit_transform(df[Importances.nlargest(int(best_col)).index])
scores = cross_val_score(stack, X, y, cv=10)
print('Stacking mean score:', pp.mean(scores), 'STD: ', pp.std(scores))

scores = cross_val_score(stack, X, y, cv=10, scoring='neg_mean_squared_error')
print('Stacking mean MSE:', pp.mean(scores), 'STD: ', pp.std(scores))

# ----------- VotingRegressor ----------- #

vote = VotingRegressor([('ExtraTrees', ExtraTrees), ('RandomForest', RandomForest), ('GB', GB),
                        ('XGBoost', XGBoost), ('LightGBM', LightGBM)])

scores = cross_val_score(vote, X, y, cv=10)
print('Voting mean:', pp.mean(scores), 'STD: ', pp.std(scores))

scores = cross_val_score(vote, X, y, cv=10, scoring='neg_mean_squared_error')
print('Voting mean MSE:', pp.mean(scores), 'STD: ', pp.std(scores))

# StackingRegressor is best clf so we will stuck with him till the end

# ----------- predicting Test Data Set  ----------- #

# Repeat all the preprocessing steps from Train DataSet
df = test
df['HasShed'] = df.MiscFeature.apply(lambda x: 1 if x == 'Shed' else 0)

df['HasTennis'] = df.MiscFeature.apply(lambda x: 1 if x == 'TenC' else 0)

df['HasGar2'] = df.MiscFeature.apply(lambda x: 1 if x == 'Gar2' else 0)

df['HasPool'] = df.PoolArea.apply(lambda x: 1 if x > 0 else 0)

df['HasDeck'] = df.WoodDeckSF.apply(lambda x: 1 if x > 0 else 0)

df['IsNew'] = df.YearBuilt.apply(lambda x: 1 if x > 2000 else 0)

df['IsOld'] = df.YearBuilt.apply(lambda x: 1 if x < 1946 else 0)

df.drop('MiscFeature', axis=1, inplace=True)

# ------------------------------- #

df['Age'] = df['YrSold'] - df['YearBuilt']

df['BsmtTotalBathRooms'] = df['BsmtFullBath'] + df['BsmtHalfBath']

df['AbvGradeTotalBathRooms'] = df['FullBath'] + df['HalfBath']

df['Total Rooms'] = df['BedroomAbvGr'] + df['BsmtFullBath'] + df['BsmtHalfBath'] + df['FullBath'] + df['HalfBath'] \
                    + df['TotRmsAbvGrd'] + df['KitchenAbvGr']

stack.fit(X, y)
test = scale.fit_transform(df[Importances.nlargest(int(best_col)).index])

pred = stack.predict(scale.fit_transform(test))

sub['SalePrice'] = pred
sub.to_csv('submission_2.csv', index=False)
# the score is around RMSE(0.3400) on Kaggle

# ------- Plot best cols ------- #
plt.figure(figsize=(20, 15))
Importances.nlargest(int(best_col)).plot(kind='barh')
plt.show()
