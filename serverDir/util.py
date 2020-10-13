import json
import pickle
import numpy as np

__locations=None
__data_columns = None
__model =None

def get_estimated_price(location,sqft,bath,bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >=0:
        x[loc_index] = 1
    return round(__model.predict([x])[0],2)

def get_all_locations():
    return __locations

def load_saved_artefacts():
    print("loading saved artefacts .......start")
    global __data_columns
    global __locations

    with open("./artefacts/bangalore_realestate_price_prediction.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    with open("./artefacts/bangalore_realestate_price_prediction.pickle",'rb') as f:
        __model = pickle.load(f)
    print("loading artefacts ... done")
if __name__ == "__main__":
    load_saved_artefacts()
    print(get_all_locations())
    print(get_estimated_price("1st block jayanagar",1000,2,3))