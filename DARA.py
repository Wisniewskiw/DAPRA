from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
from Wrapper import *
import abc
from abc import ABCMeta
import numpy as np


class ToDARA(object):
    '''
    DARA abstract class:
    '''

    __metaclass__ = ABCMeta

    def __init__(self, parameters_dict):
        '''

        :param parameters_dict: dict with many parameters
        -resolution : time granulation
        -statistics columns list : list of columns to integrate into dara format
        -open/close column names : self explanatory
        -save prefix : string for saving the final result
        '''

        if parameters_dict is None:
            parameters_dict = Wrapper({})
        else:
            parameters_dict = Wrapper(parameters_dict)

        self.resolution = parameters_dict.resolution
        self.statistics_columns_list = parameters_dict.statistics_columns_list
        self.openTimeColumnName=parameters_dict.openTimeColumnName
        self.closeTimeColumnName = parameters_dict.closeTimeColumnName
        self.save_prefix=parameters_dict.save_prefix


    def floor_resolution(self,x):
        '''

        :param x: time in milisecond for example (same dimension as resolution)
        :return: floored wrt the resolution i.e resolution 3 ,f(10)=9
        '''

        return int(x / self.resolution) * self.resolution


    def ceil_resolution(self,x):
        '''

        :param x: time in milisecond for example (same dimension as resolution)
        :return: ceiled wrt the resolution i.e resolution 3 ,f(10)=12
        '''

        tmp = int(x / self.resolution) * self.resolution

        return tmp +  self.resolution

    def make_resolution_vector_grouped(self,df_grouped):
        '''

        :param df_grouped: (grouped) datafreme i.e one may want to groupby  region ,taxi type,currency
        :return: np.array with time grid ( floor(start),floor(start) + resolution, ....,ceil(stop))
        '''

        m, M = (df_grouped[self.openTimeColumnName].min()), (df_grouped[self.closeTimeColumnName].max())
        m = self.floor_resolution(m)
        M = self.ceil_resolution(M)

        return np.arange(m, M, self.resolution)


    def indicator_and_proportions_dict_OpenCloseLockedFloat(self,res_vec,ot,ct):
        '''
        Given the resolution vector ( floor(start),floor(start) + resolution, ....,ceil(stop))
        for a given agent calculates epochtypes and proportions

        :param res_vec: resolution vector
        :param ot: open time
        :param ct: close time
        :return: dict with epoch types and proportions (lists)
        '''


        Float = 1 * ((res_vec > self.floor_resolution(ot)) * (self.floor_resolution(ct) > res_vec))
        LockedIn = 1 * ((res_vec == self.floor_resolution(ot ))
                        * (res_vec == self.floor_resolution(ct )))
        Close = (1 * ((res_vec == self.floor_resolution(ct )) )+LockedIn)%2
        Open = (1 * ((res_vec == self.floor_resolution(ot))) +LockedIn)%2

        proportionOpen = Open * 1.*(res_vec + self.resolution - ot) / (ct - ot)
        proportionFloat = Float * 1.*(self.resolution) / (ct - ot)

        proportionLockedIn = LockedIn
        proportionClose = Close * (ct - res_vec) / (ct - ot)

        out_dict={
            'Open':Open,
            'Float':Float,
            'LockedIn':LockedIn,
            'Close':Close,
            'proportionOpen':proportionOpen,
            'proportionFloat':proportionFloat,
            'proportionLockedIn':proportionLockedIn,
            'proportionClose':proportionClose
        }

        return  out_dict



    def makeDaraGroupedDataframe(self,df_grouped):
        '''
        Core of DARA :having grouped  records  (input dataframe)  tranforms into dara format

        :param df:
        :return: dataframe
        '''

        # ------------------------------------------------------------------------------------------------------
        # STEP1 :  get resolution grid vector and initialise variables
        # =======
        # ------------------------------------------------------------------------------------------------------

        x=self.make_resolution_vector_grouped(df_grouped)

        data = {'time': x
                }
        l=[self.openTimeColumnName,self.closeTimeColumnName]+self.statistics_columns_list
        l2=[df_grouped[m] for m in l]
        OpenTotal =[np.zeros(len(x)) for n in range(len(l)-1)]
        FloatTotal=[np.zeros(len(x)) for n in range(len(l)-1)]
        LockedInTotal=[np.zeros(len(x)) for n in range(len(l)-1)]
        CloseTotal=[np.zeros(len(x)) for n in range(len(l)-1)]

        # ------------------------------------------------------------------------------------------------------
        # STEP 2 :  given open ,close and other  columns of interest (statistics_columns_list),iterate row-wise
        # =======
        # ------------------------------------------------------------------------------------------------------

        for i,y  in enumerate(zip(*l2)):

            # ------------------------------------------------------------------------------------------------------
            # STEP 2a : get epoch type statistics and proportions
            # =======
            # ------------------------------------------------------------------------------------------------------

            dict = self.indicator_and_proportions_dict_OpenCloseLockedFloat(x, y[0],y[1])
            OpenTotal[0] += dict['Open']
            FloatTotal[0] += dict['Float']
            LockedInTotal[0] += dict['LockedIn']
            CloseTotal[0] += dict['Close']

            # ------------------------------------------------------------------------------------------------------
            # STEP 2b : get epoch type statistics  for other columns
            # =======
            # ------------------------------------------------------------------------------------------------------

            for j in range(2,len(l)):
                OpenTotal[j-1]+=dict['proportionOpen']*y[j]
                FloatTotal[j-1] += dict['proportionFloat'] * y[j]
                LockedInTotal[j-1] += dict['proportionLockedIn'] * y[j]
                CloseTotal[j-1] += dict['proportionClose'] * y[j]

        # ------------------------------------------------------------------------------------------------------
        # STEP 3 : construction of output data frame from dict
        # =======
        # ------------------------------------------------------------------------------------------------------
        data.update({ 'OpenTotal': OpenTotal[0], 'FloatTotal': FloatTotal[0], 'LockedInTotal': LockedInTotal[0],
                'CloseTotal': CloseTotal[0],
                'Total': OpenTotal[0] + FloatTotal[0] + LockedInTotal[0] + CloseTotal[0],
                })


        for j in range(2, len(l) ):

            data.update({'Open' + l[j]: OpenTotal[j - 1]})
            data.update({'Float' + l[j]: FloatTotal[j - 1]})
            data.update({'LockedIn' + l[j]: LockedInTotal[j - 1]})
            data.update({'Close' + l[j]: CloseTotal[j - 1]})
            data.update({'Total' + l[j]:  data['Open' +l[j]] + data['LockedIn' +l[j]] + data['Float' +l[j]] + data['Close' + l[j]]})

        out=pd.DataFrame(data)

        # ------------------------------------------------------------------------------------------------------
        # STEP 4 : eliminate NONOVERLAPING epochtype
        # =======
        # ------------------------------------------------------------------------------------------------------

        return  out[out.Total>0]


    @abc.abstractmethod
    def preprocessing(self):
        '''

        :return: preprocessed data
        '''

    @abc.abstractmethod
    def makeDaraDataframe(self):
        '''
        Perform DARA transform for all groups int the following way

        1)preprocess

        2)groupby by symbol
        for x in groupby self.grouping_columns:
            makeDaraGroupedDataframe(x)
            save


        :return:
        '''



class TaxiDARA(ToDARA):

    def __init__(self,parameters_dict):

        super(TaxiDARA, self).__init__(parameters_dict)
        parameters_dict = Wrapper(parameters_dict)

        self.grouping_columns = parameters_dict.grouping_columns
        self.df = pd.read_csv(parameters_dict.df, nrows=parameters_dict.df_nrows,parse_dates=
                            [self.closeTimeColumnName, self.openTimeColumnName], infer_datetime_format=True)
        self.preprocess = parameters_dict.preprocess
        self.is_weather = parameters_dict.is_weather

        if self.is_weather:
            self.weather_pkl=parameters_dict.weather_pkl
            self.temperature_pkl=parameters_dict.temperature_pkl
            self.humidity_pkl = parameters_dict.humidity_pkl
            self.weather_resolution= parameters_dict.weather_resolution

        if 'approx_pickup' in parameters_dict.keys():
            self.approx_pickup= parameters_dict.approx_pickup
        else:
            self.approx_pickup = 2

        if self.preprocess:

            self.pickup_longitude = parameters_dict.pickup_longitude
            self.pickup_latitude = parameters_dict.pickup_latitude
            self.dropoff_longitude = parameters_dict.dropoff_longitude
            self.dropoff_latitude = parameters_dict.dropoff_latitude


    def disc(self,longtitude, latitude, prefix):
        '''
        Rounds latitude and lontitude in order to group by region

        :param longtitude: list of longtitudes
        :param latitude: list of latitudes
        :param prefix: which disc it is pickup or drop off
        :return: list of disc coordinates
        '''

        approx_long = np.round(longtitude ,self.approx_pickup)
        approx_lat = np.round(latitude ,self.approx_pickup)

        return [prefix + '_'+str(x)+'_'+str(y) for x,y in zip(approx_long,approx_lat)]

    def add_weather(self):

        weather = pd.read_pickle(self.weather_pkl)
        weather = weather.reset_index()

        temp = pd.read_pickle(self.temperature_pkl)
        temp = temp.reset_index()

        humidity = pd.read_pickle(self.humidity_pkl)
        humidity = humidity.reset_index()

        weather['datetime'] = pd.to_datetime(weather['datetime'])
        temp['datetime'] = pd.to_datetime(temp['datetime'])
        humidity['datetime'] = pd.to_datetime(humidity['datetime'])

        d_w = dict(zip(weather.datetime.dt.floor(self.weather_resolution), weather['New York']))
        d_t = dict(zip(temp.datetime.dt.floor(self.weather_resolution), temp['New York']))
        d_h = dict(zip(humidity.datetime.dt.floor(self.weather_resolution), humidity['New York']))

        return d_w, d_t, d_h

    def preprocessing(self):
        '''
        Preprocesses the data, adds grouping discs,transforms time in seconds
        :return:
        '''

        pd.options.mode.chained_assignment = None
        from_ns_to_secs = int(1e9)

        self.df.loc[:,'DiscreteXYPickup']= self.disc(self.df[self.pickup_longitude],self.df[self.pickup_latitude],'P')
        self.df.loc[:,'DiscreteXYDropoff']= self.disc(self.df[self.dropoff_longitude], self.df[self.dropoff_latitude],'D')
        self.df[ self.openTimeColumnName] = [x.value/ from_ns_to_secs for x in self.df[self.openTimeColumnName]]
        self.df[ self.closeTimeColumnName] = [x.value / from_ns_to_secs for x in self.df[self.closeTimeColumnName]]

        pd.options.mode.chained_assignment = 'warn'
        print('-------PREPROCESSING DONE-----------')


    def makeDaraDataframe(self, which_disc = 'DiscreteXYPickup'):
        '''

        Core of DARA for this Taxi class

        :return: saved dataframe in DARA format
        '''
        # ------------------------------------------------------------------------------------------------------
        # STEP 1 : Preprocessing
        # =======
        # ------------------------------------------------------------------------------------------------------
        if self.preprocess:
            self.preprocessing()
        i=0
        # ------------------------------------------------------------------------------------------------------
        # STEP 2: Iteratively grouping by pickup location disc and  other grouping elements => save to Dara
        # =======
        # ------------------------------------------------------------------------------------------------------
        if self.is_weather:
            d_w, d_t, d_h = self.add_weather()

        for n,g  in tqdm(self.df.groupby(self.grouping_columns)):


            if  i==0:

                my_df = self.makeDaraGroupedDataframe(g)
                my_df["time"] = pd.to_datetime(my_df.time, unit='s', errors='coerce')

                if self.is_weather:
                    my_df['weather'] = my_df.time.dt.floor(self.weather_resolution).map(d_w)
                    my_df['temperature'] = my_df.time.dt.floor(self.weather_resolution).map(d_t)
                    my_df['humidity'] = my_df.time.dt.floor(self.weather_resolution).map(d_h)

                for grouping_element, grouping_name in zip(list(n), self.grouping_columns):
                    my_df[grouping_name] = grouping_element
                my_df.to_csv(self.save_prefix+'todarataxi'+str(self.resolution)+'.csv', encoding='utf-8', index=False)
                i=+1
                continue


            else:
                my_df = self.makeDaraGroupedDataframe(g)
                my_df["time"] = pd.to_datetime(my_df.time, unit='s', errors='coerce')

                if self.is_weather:
                    my_df['weather'] = my_df.time.dt.floor(self.weather_resolution).map(d_w)
                    my_df['temperature'] = my_df.time.dt.floor(self.weather_resolution).map(d_t)
                    my_df['humidity'] = my_df.time.dt.floor(self.weather_resolution).map(d_h)

                for grouping_element, grouping_name in zip(list(n), self.grouping_columns):
                    my_df[grouping_name] = grouping_element

                my_df.to_csv(self.save_prefix+'todarataxi'+str(self.resolution)+'.csv', mode='a', header=False,index=False)


class Forex(ToDARA):

    def __init__(self, parameters_dict):

        super(Forex, self).__init__(parameters_dict)
        parameters_dict = Wrapper(parameters_dict)

        self.grouping_columns = parameters_dict.grouping_columns
        self.df = pd.read_csv(parameters_dict.df, nrows=parameters_dict.df_nrows, parse_dates=
        [self.closeTimeColumnName, self.openTimeColumnName], infer_datetime_format=True)
        self.preprocess = parameters_dict.preprocess

    def preprocessing(self):
        '''
        Preprocesses the data, adds grouping discs,transforms time in seconds
        :return:
        '''

        pd.options.mode.chained_assignment = None
        from_ns_to_secs = int(1e9)

        self.df[self.openTimeColumnName] = [x.value / from_ns_to_secs for x in self.df[self.openTimeColumnName]]
        self.df[self.closeTimeColumnName] = [x.value / from_ns_to_secs for x in self.df[self.closeTimeColumnName]]

        pd.options.mode.chained_assignment = 'warn'
        print('-------PREPROCESSING DONE-----------')

    def makeDaraDataframe(self):
        '''



        :return: saved dataframe in DARA format
        '''
        # ------------------------------------------------------------------------------------------------------
        # STEP 1 : Preprocessing
        # =======
        # ------------------------------------------------------------------------------------------------------
        if self.preprocess:
            self.preprocessing()
        i = 0
        # ------------------------------------------------------------------------------------------------------
        # STEP 2: Iteratively grouping by pickup location disc and  other grouping elements => save to Dara
        # =======
        # ------------------------------------------------------------------------------------------------------

        for n, g in tqdm(self.df.groupby(self.grouping_columns)):

            if i == 0:

                my_df = self.makeDaraGroupedDataframe(g)
                my_df["time"] = pd.to_datetime(my_df.time, unit='s', errors='coerce')

                for grouping_element, grouping_name in zip(list(n), self.grouping_columns):
                    my_df[grouping_name] = grouping_element

                my_df.to_csv(self.save_prefix + 'todaraforex' + str(self.resolution) + '.csv', encoding='utf-8',
                             index=False)
                i = +1
                continue


            else:
                my_df = self.makeDaraGroupedDataframe(g)
                my_df["time"] = pd.to_datetime(my_df.time, unit='s', errors='coerce')

                for grouping_element, grouping_name in zip(list(n), self.grouping_columns):
                    my_df[grouping_name] = grouping_element

                my_df.to_csv(self.save_prefix + 'todaraforex' + str(self.resolution) + '.csv', mode='a', header=False,
                             index=False)








