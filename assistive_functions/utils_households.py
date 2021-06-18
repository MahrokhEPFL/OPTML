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
    candidates = pd.read_csv(path)
    # filter by group
    if 'group' in household_options.keys():
        candidates = candidates.loc[data.Acorn==household_options['group']]
    else:
        print('[INFO] no filtering by group')
    # filter by tariff 
    candidates = candidates.loc[candidates.stdorToU==household_options['stdorToU']]
    # print(candidates)    
    # TODO: shuffle
    households=[]
    step_ahead=1

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
        # load data 
        has_data = household.load_data()
        if has_data:
            households.append(household)
            needed = needed-1
        else:
            print('[INFO] households discarded\n')
        # search next
        num = num+1

    print('\n[INFO] Connected to ' + str(len(households)) + ' households')
    return households

