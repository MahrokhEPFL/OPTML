import os
import sys
import copy
import torch
import numpy as np
import pandas as pd

# candidate lags
def get_lags(step_ahead, num_days=np.array([0,1,7])):
    lags = []
    for i in num_days:
        lags = lags + [48*i, 48*i+1, 48*i+2]
    lags = [x-step_ahead+1 for x in lags if x>=step_ahead]
    return lags

def connect_to_households(household_options):
    from household import Household
    # get candidate houses from the selected group
    path = os.getcwd()+"/input/informations_households.csv.xls"
    data = pd.read_csv(path)
    # filter by group
    candidates = data.loc[data.Acorn==household_options['group']]
    # filter by tariff 
    candidates = candidates.loc[candidates.stdorToU==household_options['stdorToU']]
    # print(candidates)    
    # TODO: shuffle
    households=[]
    step_ahead=1

    # default options (only to check if the household has enough data)
    def_options = {"dayparts":[],
           "resolution":60,
           "remove_holiday":True,
           "filt_days":['Tuesday'], 
           "replacement_method":'week_before',
           "feat_cols":['hourofd_x', 'hourofd_y', 'dayofy_x', 'dayofy_y', 'temperature_hourly']}

    # create households
    needed = household_options['num_households']
    num = 0
    while needed>0:
        # check if there are enough households
        if num>=len(candidates):
            num_households = len(households)
            print('[Warning] could not find enough households')
            print('[Warning] changed number of households to ' + str(num_households))
        # get household
        household = Household(house_id=candidates.LCLid.iloc[num],
                               block_num=candidates.file.iloc[num])
        # load data with default options
        household.construct_dataset(lags=[1], step_ahead=step_ahead, options=def_options)
        if len(household.y) > 0:
            households.append(household)
            needed = needed-1
        # search next
        num = num+1

    print('\n[INFO] Connected to ' + str(len(households)) + ' households')
    return households

def penalty(model_n, model_0, lambda_):
    st_dict_n = copy.deepcopy(model_n.state_dict())
    st_dict_0 = copy.deepcopy(model_0.state_dict())
    w_n = np.hstack((st_dict_n['linear.bias'].numpy(),st_dict_n['linear.weight'].numpy().flatten()))
    w_0 = np.hstack((st_dict_0['linear.bias'].numpy(),st_dict_0['linear.weight'].numpy().flatten()))
    return lambda_ * torch.nn.MSELoss()(torch.FloatTensor(w_n), torch.FloatTensor(w_0))