from DARA import *

def preprocess_clean(path_trip_data , path_fare_data, index=1,add_weather=False,
                     weather_path_dict={'weather':"weather.pkl",
                                         'temp':"temp.pkl",
                                         'humidity':'humidity.pkl',
                                         'time_resolution':'H'}
                     ):
    '''

    :param path_trip_data:
    :param path_fare_data:
    :return: cleaned  taxi data
    '''

    #load files
    print('-----------CSVs LOADING---------')

    df2=pd.read_csv(path_trip_data )
    df3=pd.read_csv(path_fare_data )

    print('-----------MEARGING  & CLEANING---------')

    #renaming columns (at least i found this issue in january file )and getting rid off bad localisations
    df3.rename(columns={' hack_license': 'hack_license', ' pickup_datetime': 'pickup_datetime'}, inplace=True)
    df4 = pd.merge(df2, df3, on=['medallion', 'hack_license', 'pickup_datetime'])
    del df2,df3


    df4 = df4[['medallion', 'hack_license', 'vendor_id', 'rate_code',
               'pickup_datetime', 'dropoff_datetime',
               'passenger_count', 'trip_time_in_secs', 'trip_distance',
               'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
               'dropoff_latitude', ' vendor_id', ' payment_type', ' fare_amount',
               ' surcharge', ' mta_tax', ' tip_amount', ' tolls_amount',
               ' total_amount']]
    df4 = df4[~df4.isnull().any(axis=1)]

    df4 = df4[
              (df4['trip_time_in_secs'] > 0) &
              (df4['trip_distance'] > 0) & (df4[' total_amount'] > 0) &
              (df4['dropoff_longitude'] > -79.5) & (df4['dropoff_longitude'] < -71) &
              (df4['dropoff_latitude'] > 40) & (df4['dropoff_latitude'] < 45) &
              (df4['pickup_longitude'] > -79.5) & (df4['pickup_longitude'] < -71) &
              (df4['pickup_latitude'] > 40) & (df4['pickup_latitude']<45)
            ]
    df4 = df4[df4.pickup_datetime != df4.dropoff_datetime]
   

    if add_weather:
        print('-----------ADDING WEATHER---------')
        #weather load
        weather = pd.read_pickle(weather_path_dict['weather'])
        weather = weather.reset_index()
        temp = pd.read_pickle(weather_path_dict['temp'])
        temp = temp.reset_index()
        humidity = pd.read_pickle(weather_path_dict['humidity'])
        humidity = humidity.reset_index()
        weather['datetime'] = pd.to_datetime(weather['datetime'])
        temp['datetime'] = pd.to_datetime(temp['datetime'])
        humidity['datetime'] = pd.to_datetime(humidity['datetime'])

        #weather join
        df4['pickup_datetime'] = pd.to_datetime(df4['pickup_datetime'])
        d = dict(zip(weather.datetime.dt.floor(weather_path_dict['time_resolution']), weather['New York']))
        df4['weather'] = df4.pickup_datetime.dt.floor(weather_path_dict['time_resolution']).map(d)
        d = dict(zip(temp.datetime.dt.floor(weather_path_dict['time_resolution']), temp['New York']))
        df4['temperature'] = df4.pickup_datetime.dt.floor(weather_path_dict['time_resolution']).map(d)
        d = dict(zip(humidity.datetime.dt.floor(weather_path_dict['time_resolution']), humidity['New York']))
        df4['humidity'] = df4.pickup_datetime.dt.floor(weather_path_dict['time_resolution']).map(d)


    print('-----------SAVING :' +'cleaned_taxi_' + str(index) + '.csv'+'---------')

    df4.to_csv('cleaned_taxi_' + str(index) + '.csv', index=False)#slow pd.to_hdf(r'path/file.h5', mode='w')


def  preprocess_clean_randomtaxi(path_data ):
    '''

    :param path_data:
    :return: cleaned taxi data
    '''

    taxi = pd.read_csv(path_data,parse_dates=['StartRide','EndRide'], infer_datetime_format=True)
    taxi = taxi[taxi.StartRide < taxi.EndRide]
    taxi.to_csv('Random_taxi_cleaned.csv', encoding='utf-8', index=False)
    print('-----------DONE---------')







