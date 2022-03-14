import time

from sklearn import metrics
import pandas as pd
import config
import joblib
import seaborn as sb
import matplotlib.pyplot as plt
import argparse
import warnings
import numpy as np
import model_dispatcher
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')


sub = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\sample_submission.csv')
test = pd.read_csv(config.processed_TEST_FILE)
df = pd.read_csv(config.processed_TRAINING_FILE).sample(frac=1, random_state=0)

drop_f = ['passenger_count', 'pickup_longitude', 'pickup_latitude']
df.drop(drop_f, axis=1, inplace=True)
''''lower_limit = df['trip_duration'].quantile(0.1)
upper_limit = df['trip_duration'].quantile(0.99)

df = df[(df['trip_duration'] > lower_limit) & (df['trip_duration'] < upper_limit)].reset_index(drop=True)'''
test.drop(drop_f, axis=1, inplace=True)
print('--------------------------------------------')
df['kfold'] = -1
kf = KFold()
for fold, (tr, val) in enumerate(kf.split(df)):
    df.loc[val, 'kfold'] = fold


def run(fold, model):
    # read data
    train = df[df.kfold != fold].reset_index(drop=True)
    x_train = train.drop(['trip_duration', 'kfold'], axis=1)
    im = pd.Series(x_train.columns)
    y_train = train['trip_duration'].values

    valid = df[df.kfold == fold].reset_index(drop=True)
    x_valid = valid.drop(['trip_duration', 'kfold'], axis=1)
    y_valid = valid['trip_duration'].values

    print(f'train shape {x_train.shape} ; valid shape {x_valid.shape} ')
    clf = model_dispatcher.models[model]

    start = time.time()
    clf.fit(x_train, y_train)

    valid_pred = clf.predict(x_valid)

    train_pred = clf.predict(x_train)

    train_score = np.sqrt(metrics.mean_squared_error(y_train, train_pred))
    valid_score = np.sqrt(metrics.mean_squared_error(y_valid, valid_pred))

    sub['trip_duration'] = np.exp(clf.predict(test))
    sub.to_csv(fr'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\fold_{fold}_pred_{np.round(valid_score, 4)}.csv', index=False)

    sub.to_csv(rf'sub_fold_{fold}_pred_{np.round(valid_score, 4)}_last.csv', index=False)

    end = time.time()

    print(
        f'train_score: {train_score:5f}, valid_score: {valid_score:5f}, fold: {fold}, time: {np.round((end - start) / 60, 2)}:min')

    sb.barplot(sorted(clf.feature_importances_), im.values)
    plt.show()
# joblib.dump(clf,f'{config.MODEL_OUTPUT}_{fold}.bin')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    run(fold=args.fold, model=args.model)

