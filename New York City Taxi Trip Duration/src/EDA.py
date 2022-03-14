import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import config
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 2000)


def date(data, col):
    data[col] = pd.to_datetime(data[col])

    data['month'] = data[col].dt.month
    data['week'] = data[col].dt.week
    data['weekday'] = data[col].dt.weekday
    data.drop(col, axis=1, inplace=True)


'''for col in weather:
    if weather[col].nunique() < 40:
        sb.countplot(weather[col])
    else:
        sb.displot(weather[col])
    plt.title(f'col: {col}')
    plt.show()'''
df = pd.read_csv(config.processed_TRAINING_FILE)
test = pd.read_csv(config.processed_TEST_FILE)

date(weather, 'date')
weather = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\weather_data_nyc_centralpark_2016(1).csv')

full_df = pd.merge(df, weather, how='outer', on=['week', 'month', 'weekday']).drop(['precipitation', 'snow fall', 'snow depth'], axis=1).dropna()

full_test = pd.merge(test, weather, how='outer', on=['week', 'month', 'weekday']).drop(['precipitation', 'snow fall', 'snow depth'], axis=1).dropna()

# full_df.to_csv(f'weather_processed_TRAINING_FILE.csv', index=False)
# full_test.to_csv(f'weather_processed_TEST_FILE.csv', index=False)


# ---outliers---#
'''
print(f': length: {len(df)}, std: {df["trip_duration"].std()}')

# Z-score:
upper_limit = df['trip_duration'].mean() + 3 * df['trip_duration'].std()
lower_limit = df['trip_duration'].mean() - 3 * df['trip_duration'].std()

z_score = df[(df['trip_duration'] > lower_limit) & (df['trip_duration'] < upper_limit)]

print(f'z-score: length: {len(z_score)}, std: {z_score["trip_duration"].std()}')

# IQR:
percentile25 = df['trip_duration'].quantile(0.25)
percentile75 = df['trip_duration'].quantile(0.75)
IQR = percentile75 - percentile25

lower_limit = percentile25 - 1.5 * IQR
upper_limit = percentile75 + 1.5 * IQR
IQR = df[(df['trip_duration'] > lower_limit) & (df['trip_duration'] < upper_limit)]
print(f'IQR: length: {len(IQR)}, std: {IQR["trip_duration"].std()}')

# percentage
lower_limit = df['trip_duration'].quantile(0.1)
upper_limit = df['trip_duration'].quantile(0.99)
percentage = df[(df['trip_duration'] > lower_limit) & (df['trip_duration'] < upper_limit)]
print(f'percentage: length: {len(percentage)}, std: {percentage["trip_duration"].std()}')
plt.subplot(411)
sb.boxplot(df['trip_duration'])

plt.subplot(412)
sb.boxplot(z_score['trip_duration'])


plt.subplot(413)
sb.boxplot(IQR['trip_duration'])

plt.subplot(414)
sb.boxplot(percentage['trip_duration'])
plt.show()


'''