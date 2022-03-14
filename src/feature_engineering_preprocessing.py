import pandas as pd
import numpy as np
import config, warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans

pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 2000)

train = pd.read_csv(config.TRAINING_FILE)
test = pd.read_csv(config.TESTING_FILE)


# Function aiming at calculating the direction
def ft_degree(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def ft_haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def test_engineer_features(data, save_to=r'../inputs/test_processed_file.csv'):
    # date features
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

    data['month'] = data.pickup_datetime.dt.month
    data['week'] = data.pickup_datetime.dt.week
    data['weekday'] = data.pickup_datetime.dt.weekday
    data['day'] = data.pickup_datetime.dt.day
    data['hour'] = data.pickup_datetime.dt.hour
    data['minute'] = data.pickup_datetime.dt.minute
    data['minute_oftheday'] = data['hour'] * 60 + data['minute']

    # combine weather data:
    weather = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\weather_data_nyc_centralpark_2016(1).csv')
    weather['date'] = pd.to_datetime(weather['date'])
    weather['month'] = weather.date.dt.month
    weather['week'] = weather.date.dt.week
    weather['weekday'] = weather.date.dt.weekday
    weather['day'] = weather.date.dt.day
    data = pd.merge(data, weather, how='outer', on=['week', 'month', 'weekday', 'day']).drop(
        ['precipitation', 'snow fall', 'snow depth', 'date'], axis=1).dropna()


    data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map({'Y': 1, 'N': 0})
    # ----------------------------------------#
    # Distance on earth
    data['space_distance'] = ft_haversine_distance(data['pickup_latitude'].values,
                                                   data['pickup_longitude'].values,
                                                   data['dropoff_latitude'].values,
                                                   data['dropoff_longitude'].values)

    # magnitude(speed m/s) coords: √[(x₂ - x₁)² + (y₂ - y₁)²]
    data['dist_long'] = data['dropoff_longitude'] - data['pickup_longitude']
    data['dist_lat'] = data['dropoff_latitude'] - data['pickup_latitude']
    data['magnitude(speed m/s)'] = np.sqrt(np.square(data['dist_long']) + np.square(data['dist_lat']))
    # ----------------------------------------#
    # Add direction feature
    data['direction'] = ft_degree(data['pickup_latitude'].values,
                                  data['pickup_longitude'].values,
                                  data['dropoff_latitude'].values,
                                  data['dropoff_longitude'].values)

    # ----------------------------------------#
    data.drop(['dist_long', 'dist_lat', 'id', 'pickup_datetime', 'vendor_id'],
              axis=1, inplace=True)

    # ----------------------------------------#
    # create region
    kmeans = pd.read_pickle(r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\region_coords_cluster.pkl')
    data.loc[:, 'pickup_cluster'] = kmeans.predict(data[['pickup_latitude', 'pickup_longitude']])
    data.loc[:, 'dropoff_cluster'] = kmeans.predict(data[['dropoff_latitude', 'dropoff_longitude']])

    # create sub-region
    kmeans = pd.read_pickle(
        r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\sub_region_coords_cluster.pkl')
    data.loc[:, 'sub_pickup_cluster'] = kmeans.predict(data[['pickup_latitude', 'pickup_longitude']])
    data.loc[:, 'sub_dropoff_cluster'] = kmeans.predict(data[['dropoff_latitude', 'dropoff_longitude']])

    # scaler = pd.read_pickle('../inputs/feature_scaler.pkl')
    # data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    data.to_csv(f'{save_to}', index=False)


import seaborn as sb
import matplotlib.pyplot as plt


def train_engineer_features(data, save_to=r'../inputs/engineered_file.csv'):
    # date features
    data = data[(data.trip_duration < 5900)]

    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

    data['month'] = data.pickup_datetime.dt.month
    data['week'] = data.pickup_datetime.dt.week
    data['weekday'] = data.pickup_datetime.dt.weekday
    data['day'] = data.pickup_datetime.dt.day
    data['hour'] = data.pickup_datetime.dt.hour
    data['minute'] = data.pickup_datetime.dt.minute
    data['minute_oftheday'] = data['hour'] * 60 + data['minute']

    # combine weather data:
    weather = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\weather_data_nyc_centralpark_2016(1).csv')
    weather['date'] = pd.to_datetime(weather['date'])
    weather['month'] = weather.date.dt.month
    weather['week'] = weather.date.dt.week
    weather['weekday'] = weather.date.dt.weekday
    weather['day'] = weather.date.dt.day
    data = pd.merge(data, weather, how='outer', on=['week', 'month', 'weekday', 'day']).drop(
        ['precipitation', 'snow fall', 'snow depth','date'], axis=1).dropna()
    # ----------------------------------------#

    # Distance on earth
    data['space_distance'] = ft_haversine_distance(data['pickup_latitude'].values,
                                                   data['pickup_longitude'].values,
                                                   data['dropoff_latitude'].values,
                                                   data['dropoff_longitude'].values)

    # distance coords: √[(x₂ - x₁)² + (y₂ - y₁)²]
    data['dist_long'] = data['dropoff_longitude'] - data['pickup_longitude']
    data['dist_lat'] = data['dropoff_latitude'] - data['pickup_latitude']
    data['coords_distance'] = np.sqrt(np.square(data['dist_long']) + np.square(data['dist_lat']))
    # ----------------------------------------#
    # Add direction feature
    data['direction'] = ft_degree(data['pickup_latitude'].values,
                                  data['pickup_longitude'].values,
                                  data['dropoff_latitude'].values,
                                  data['dropoff_longitude'].values)

    data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map({'Y': 1, 'N': 0})
    # ----------------------------------------#
    data.drop(['dist_long', 'dist_lat', 'id', 'pickup_datetime', 'dropoff_datetime', 'vendor_id'],
              axis=1, inplace=True)

    # ----------------------------------------#
    # OUTLIERS

    X = data.drop('trip_duration', axis=1)

    # drop passenger_count outlier
    X = X.loc[(X['passenger_count'] <= 7) & (X['passenger_count'] > 0)]

    # drop coordinates outlier
    '''X = X.loc[(X['dropoff_latitude'].values > 40.6) & (X['dropoff_latitude'].values < 40.9)]
    X = X.loc[(X['pickup_latitude'] > 40.6) & (X['pickup_latitude'] < 40.9)]

    X = X.loc[(X['dropoff_longitude'] > -74.06) & (X['dropoff_longitude'] < -73.6)]
    X = X.loc[(X['pickup_longitude'] > -74.06) & (X['pickup_longitude'] < -73.6)]'''
    X = X[(X.pickup_longitude > -100)]
    X = X[(X.pickup_latitude < 50)]

    # drop false coordinates

    ''''false_coords = X.loc[(X['pickup_latitude'].values == X['dropoff_latitude'].values) &
                         (X['pickup_longitude'].values == X['dropoff_longitude'].values)].index
    X.drop(index=false_coords, axis=0, inplace=True)'''
    # ---------#
    coords = np.vstack((X[['pickup_latitude', 'pickup_longitude']].values,
                        X[['dropoff_latitude', 'dropoff_longitude']].values))

    # create region
    kmeans = MiniBatchKMeans(100)
    kmeans.fit(coords)
    pd.to_pickle(kmeans, r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\region_coords_cluster.pkl')

    X.loc[:, 'pickup_cluster'] = kmeans.predict(X[['pickup_latitude', 'pickup_longitude']])
    X.loc[:, 'dropoff_cluster'] = kmeans.predict(X[['dropoff_latitude', 'dropoff_longitude']])

    # create sub-region

    kmeans = MiniBatchKMeans(1000, random_state=999)
    kmeans.fit(coords)

    pd.to_pickle(kmeans,
                 r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\sub_region_coords_cluster.pkl')

    X.loc[:, 'sub_pickup_cluster'] = kmeans.predict(X[['pickup_latitude', 'pickup_longitude']])
    X.loc[:, 'sub_dropoff_cluster'] = kmeans.predict(X[['dropoff_latitude', 'dropoff_longitude']])

    # processing
    scaler = MinMaxScaler()

    scaler.fit(X)
    # df = pd.DataFrame(scaler.fit_transform(df, columns=df.columns)
    pd.to_pickle(scaler, r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\feature_scaler.pkl')

    y = np.log(data['trip_duration'].values)
    data = pd.concat([X, pd.Series(y, name='trip_duration')], axis=1).dropna()

    data.to_csv(f'{save_to}', index=False)


train_engineer_features(train, config.processed_TRAINING_FILE)
test_engineer_features(test, config.processed_TEST_FILE)
