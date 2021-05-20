from load_data import get_data_of_a_person
from sklearn.model_selection import train_test_split
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def add_lags(df_1p, lags, feat_cols, step_ahead=1, replacement_method='week_before'):
    '''
    step_ahead: 
    methods:
        - "ignore": do nothing
        - "remove": fill with nan
        - "day_before": replace with 1 day ago
        - "day_before_it": replaces with the closest day that was not a holiday or weekend
        - "week_ before": replace with 1 week ago
    '''
    if not replacement_method == 'ignore':
        # holiday indexes
        hol_inds = df_1p.index[df_1p.flg_holiday].values
        # assign replacement lag
        if replacement_method=='day_before' or replacement_method=='day_before_it':
            rep_lag_init = 48
        if replacement_method=='week_before' or replacement_method=='week_before_it':    
            rep_lag_init = 48*7
    
    # add weather and temporal features to regression df
    max_ind = df_1p.energy.size
    max_lag = max(lags)
    cols = feat_cols + ['flg_holiday', 'weekday', 'month', 'daypart', 'season']
    df_reg = df_1p.loc[:,cols].iloc[max_lag:max_ind-step_ahead+1,:]
    
    # add target to regression df
    df_reg['target'] = df_1p.energy.iloc[max_lag+step_ahead-1:max_ind].values
    for lag in lags:
        col_name = 'lag ' + str(lag)
        df_reg[col_name] = df_1p.energy.iloc[max_lag-lag:max_ind-lag-step_ahead+1].values
    
    # Treat holidays
    if not replacement_method == 'ignore':
        for lag in lags:
            col_name = 'lag ' + str(lag)
            #print(col_name)
            
            # if replacement is already in the lags, go twice before
            rep_lag = rep_lag_init
            while rep_lag in lags:
                rep_lag = rep_lag_init + rep_lag
                
            # where holidays appear in this column  
            hol_inds_here = hol_inds-(max_lag-lag) 
            hol_inds_here = hol_inds_here[(hol_inds_here>=0) & (hol_inds_here<df_reg.target.size)]
            #print('hol_inds_here ' + str(hol_inds_here))
            if len(hol_inds_here)>0:
                can_replace_ind = hol_inds_here[hol_inds_here+max_lag-lag>=rep_lag]
                # replacement values
                if len(can_replace_ind)>0:
                    rep_val = df_1p.energy.iloc[can_replace_ind+(max_lag-lag)-rep_lag].values
                    #print('rep_val ' + str(rep_val))
                    df_reg[col_name].iloc[can_replace_ind]=rep_val
    
    df_reg = df_reg.reset_index(drop=True)
    return df_reg


 # *************** Filter *************** # 
def filter_dataset(df_reg, filt_days, remove_holiday, dayparts=[]):
    # select only given weekdays
    df_reg_filt = df_reg.loc[df_reg['weekday'].isin(filt_days), :]
    # remove holidays
    if remove_holiday:
        df_reg_filt = df_reg_filt.loc[df_reg_filt.flg_holiday==False, :]
    # select dayparts if specified
    if not dayparts==[]:
        df_reg_filt = df_reg_filt.loc[df_reg['daypart'].isin(dayparts), :]
    # reset index
    df_reg_filt.reset_index(drop=True,inplace=True)
    return df_reg_filt


# *************** Split *************** # 
def split(df, shuffle, test_size, feat_cols, verbose=False):
    y = df.target.values
    feat_names = feat_cols + [col for col in df if col.startswith('lag')]
    X = df.loc[:,feat_names]
    feat_names = X.columns.tolist()
    if verbose:
        print('Features ' + str(X.columns.tolist()))

    if shuffle:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        ind = int(len(y)*(1-test_size))
        X_train = X.loc[0:ind,:]
        y_train = y[0:ind]
        X_test = X.loc[ind:,:]
        y_test = y[ind:]
    # get train and test indexes
    df['istest'] = [False]*len(df.target)
    test_ind = X_test.index.to_list()
    df.istest.iloc[test_ind]=True
    if verbose:
        print('Training size = ' + str(len(y_train)) + ' , test size = '+ str(len(y_test)))
    return df, X_train, X_test, y_train, y_test, feat_names

    
 # *************** Construct *************** # 
def construct_dataset(df_1p,
                      step_ahead = 1,
                      normalize_input=True, 
                      filt_days = ['Tuesday'],
                      remove_holiday=True,
                      lags = [1,2,48,49,7*48,7*48+1],
                      feat_cols = ['hourofd_x', 'hourofd_y', 'dayofy_x', 'dayofy_y', 'temperature_hourly'],
                      replacement_method='week_before',
                      dayparts=[]):
    # add lags
    df_reg = add_lags(df_1p, lags=lags, step_ahead=step_ahead,
                      feat_cols=feat_cols, 
                      replacement_method=replacement_method)
    # filter
    df_reg = filter_dataset(df_reg,
                            filt_days = filt_days, 
                            remove_holiday=remove_holiday,
                            dayparts=dayparts)
    # normalize
    if normalize_input:
        df_reg=normalize(df_reg)
    # get matrices
    y = df_reg.target.values
    feat_names = feat_cols + [col for col in df_reg if col.startswith('lag')]
    X = df_reg.loc[:,feat_names]
    feat_names = X.columns.tolist()
    return df_reg, X, y, feat_names


def normalize(df):
    result = df.copy()
    all_cols = df.columns
    cols=[]  
    for c in all_cols:
        if c == 'temperature_hourly':
            cols = ['temperature_hourly']
            break
   
    cols = cols + [x for x in all_cols if x.startswith('lag')]
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name].values - min_value) / (max_value - min_value)
    return result
#test
#n=60
#flg_holiday = [False] * n
#flg_holiday[1] = True
#flg_holiday[53] = True
#a = pd.DataFrame({'flg_holiday': flg_holiday ,'energy':np.arange(0,n), 'temperature':np.arange(0,n)})
#construct_dataset(a, lags=[1,2,10], weather_cols=['temperature'], temporal_cols=['flg_holiday'], replacement_method='day_before')