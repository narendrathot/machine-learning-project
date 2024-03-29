import json
import pickle
import numpy as np

__locations=None
__data_columns=None
__model_=None


def get_location_names():
    return __locations
def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        return -1


    d = np.zeros(len(__data_columns))
    d[0] = sqft
    d[1] = bath
    d[2] = bhk
    if loc_index >= 0:
        d[loc_index] = 1

    return round(__model_.predict([d])[0])



def load_data():
    global __locations, __data_columns
    with open("./server/artifacts/columns.json",'r') as f:
        __data_columns=json.load(f)['data_columns']
        __locations=__data_columns[3:0]

    global  __model_
    with open("./server/artifacts/banglore_home_prices_model.pickle",'rb') as f:
        __model_=pickle.load(f)
        print('loading saved model id done')




if  __name__ == "__main__":
    load_data()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 2))