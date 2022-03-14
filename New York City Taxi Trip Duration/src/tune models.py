import time
import pandas as pd
import config
import argparse
import warnings

warnings.filterwarnings('ignore')
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold

df = pd.read_csv(config.processed_TRAINING_FILE).sample(frac=1, random_state=0)

import skopt
import skopt.space as space
from skopt.utils import use_named_args

# SVR(C, gamma, kernel, degree)
search_space = [space.Categorical(['linear', 'rbf', 'sigmoid'], name='kernel'),
                space.Real(0.0001, 100.0, prior='log-uniform', name='C'),
                space.Real(0.0001, 100.0, prior='log-uniform', name='gamma'),
                space.Integer(1, 5, name='degree')]

# Ridge(alpha=99.98911327353338)
'''
Best Accuracy: 0.7923868967144851
Best Parameters: [99.98911327353338]
Total hours:  0.08'''
search_space_2 = [space.Real(0.0001, 100.0, prior='log-uniform', name='alpha')]

# LGBMRegressor
search_space_3 = [space.Real(0.00001, 2.0, prior='uniform', name='learning_rate'),
                space.Real(0.1, 1.0, prior='uniform', name='subsample'),
                space.Integer(1, 100, name='max_depth'),
                space.Integer(20, 100, name='num_leaves'),
                space.Integer(1, 2000, name='n_estimators'),
                ]
'''
Best Accuracy: 0.7923815818876764
Best Parameters: [1e-05, 0.19085248750566444, 21, 80, 2000]
Total hours:  0.89'''

from sklearn.ensemble import RandomForestRegressor
search_space_4 = [space.Real(0.01, 1.0, prior='uniform', name='max_features'),
                space.Integer(1, 100, name='max_depth'),
                space.Categorical([True, False], name='oob_score'),
                space.Integer(0, 100, name='random_state'),
                space.Integer(40, 500, name='min_samples_leaf'),
                space.Integer(1, 3000, name='n_estimators'),
                ]

def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))
from sklearn.linear_model import Ridge

start = time.time()


@use_named_args(search_space_4)
def evaluate_model(**params):
    mean_score = []
    for fold, (tr, val) in enumerate(KFold(n_splits=3).split(df)):
        train = df.loc[tr].reset_index(drop=True)
        x_train = train.drop(['trip_duration'], axis=1).values
        y_train = train['trip_duration']

        valid = df.loc[val].reset_index(drop=True)
        x_valid = valid.drop(['trip_duration'], axis=1).values
        y_valid = valid['trip_duration']

        clf = RandomForestRegressor()

        clf.set_params(**params)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_valid)

        y_valid = np.exp(y_valid)
        pred = np.exp(pred)

        score = rmsle(y_valid, pred)
        mean_score.append(score)
    return np.mean(mean_score)


result = skopt.gp_minimize(evaluate_model, search_space_4 ,verbose=True, n_calls=50)
# summarizing finding:
print(f'Best Accuracy: {result.fun}')
print(f'Best Parameters: {result.x}')
end = time.time()
print('Total hours: ', np.round(((end - start) / 3600), 2))
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
plot_convergence(result)
plt.show()
'''if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    run(fold=args.fold, model=args.model)
'''
