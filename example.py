import json
from funs_for_example import  *
'''
EXAMPLE of DATA  abstract class  extended to TAXI
All necessary parameters are in 2 json files
DARA works exactly like in David's example for random taxi 

Below two examples for real and fake data
Bare in mind first one need to clean and save the raw data and then lunch TaxiDARA
 
'''
if __name__ == "__main__":

    preprocess_clean_randomtaxi('RandomTaxi.csv')           #Davids RANDOM TAXI
    #preprocess_clean('trip_data_1.csv' ,'trip_fare_1.csv') #REAL TAXI 1st month

    with open('RandomTaxi.json', 'r') as fp:
        parameters_dict_random_taxi = json.load(fp)

    #with open('cleaned_taxi_1.json', 'r') as fp:
    #    parameters_dict_real_taxi = json.load(fp)



    TAXI = TaxiDARA(parameters_dict_random_taxi)
    TAXI.makeDaraDataframe()

    #TAXI = TaxiDARA(parameters_dict_real_taxi)
    #TAXI.makeDaraDataframe()


