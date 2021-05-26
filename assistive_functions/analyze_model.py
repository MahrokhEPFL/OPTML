import colorsys
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from statsmodels.sandbox.regression.predstd import wls_prediction_std

    
# ************************* PLOT 2 ************************* # 
def residuals_plot(df_reg_filt, model, feat_names, on_x_axis = 'pred'): 
    # mark train and test data
    istest = [False]*len(df_reg_filt.target.values)
    for ind in model.test_inds_:
        istest[ind] = True
    df_reg_filt['istest'] = pd.Series(istest)
    
    # setting plot style
    plt.style.use('fivethirtyeight')
    color_train = 'orange'
    color_test = 'skyblue'
    color_dict = {
        "spring": "cyan",
        "summer": "orange",
        "autumn":"darkorchid",
        "winter": "teal"}

    # add prediction
    X = df_reg_filt.loc[:,feat_names].values
    df_reg_filt['pred'] = model.predict(X)
    
    num=0
    fig, axs = plt.subplots(3,2,figsize=(6*3,6*2))
    xmin = min(df_reg_filt.loc[:,on_x_axis])
    xmax = max(df_reg_filt.loc[:,on_x_axis])
    # plot different day parts
    for daypart in df_reg_filt.daypart.unique():
        ax=axs[int(num/2),num%2]
        num=num+1
        # filter
        df_temp = df_reg_filt.loc[df_reg_filt.daypart==daypart,:]
        # train test split
        y_train = df_temp.target.loc[df_temp.istest==False].values
        y_pred_train = df_temp.pred.loc[df_temp.istest==False].values
        y_test = df_temp.target.loc[df_temp.istest].values
        y_pred_test = df_temp.pred.loc[df_temp.istest].values
        
        # plotting residual errors in training data
        if on_x_axis == 'target':
            ax.scatter(y_train, y_pred_train - y_train,
                        color = color_train, s = 10, label = 'Train data') 
            # plotting residual errors in test data
            ax.scatter(y_test, y_pred_test - y_test,
                        color = color_test, s = 10, label = 'Test data') 
            ax.set_xlabel('Actual')
        else:
            ax.scatter(y_pred_train, y_pred_train - y_train,
                        color = "green", s = 10, label = 'Train data') 
            # plotting residual errors in test data
            ax.scatter(y_pred_test, y_pred_test - y_test,
                        color = "blue", s = 10, label = 'Test data') 
            ax.set_xlabel('Predicted')
            
        # plotting line for zero residual error
        ax.hlines(y = 0, xmin = xmin, xmax = xmax, linewidth = 2)
        # labels
        ax.set_ylabel('Residual (predicted-actual)')
        ax.legend(loc = 'upper right')
        ax.set_title("Residual errors - " + daypart)
    # all
    ax = axs[2,1]
    # plotting line for zero residual error
    ax.hlines(y = 0, xmin = xmin, xmax = xmax, linewidth = 2)
    for season in df_reg_filt.season.unique():
        df_temp = df_reg_filt.loc[df_reg_filt.season==season,:]
        # plotting residual errors in training data
        if on_x_axis == 'target':
            ax.scatter(df_temp.target.values, df_temp.pred.values - df_temp.target.values,
                        color = color_dict[season], s = 10, label = season) 
            ax.set_xlabel('Actual')
        else:
            ax.scatter(df_temp.pred.values, df_temp.pred.values - df_temp.target.values,
                        color = color_dict[season], s = 10, label = season) 
            ax.set_xlabel('Predicted')
            
        ax.set_title("Residual errors - seasons" )
        ax.set_ylabel('Residual (predicted-actual)')
        ax.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()
    
    
    
def plot_actual_prediction(df_reg_filt, model, feat_names):
    # mark train and test data
    istest = [False]*len(df_reg_filt.target.values)
    for ind in model.test_inds_:
        istest[ind] = True
    df_reg_filt['istest'] = pd.Series(istest)

    # setting plot styles
    plt.style.use('fivethirtyeight')
    color_train = 'orange'
    color_test = 'skyblue'
    color_dict = {
        "spring": "cyan",
        "summer": "orange",
        "autumn":"darkorchid",
        "winter": "teal"}
    # add prediction
    X = df_reg_filt.loc[:,feat_names].values
    df_reg_filt['pred'] = model.predict(X)
    
    
    # ************************* PLOT 1 ************************* #
    # Plot Actual vs. Linear Regression predicted usage.
    fig, axs = plt.subplots(3,2,figsize=(6*3,6*2))
    num=0
    for daypart in df_reg_filt.daypart.unique():
        ax=axs[int(num/2),num%2]
        num=num+1
        df_temp = df_reg_filt.loc[df_reg_filt.daypart==daypart,:]
        ax.plot([0,5], [0,5], c='k')
        # scatter
        ax.scatter(df_temp.target.loc[df_temp.istest].values, 
                   df_temp.pred.loc[df_temp.istest].values, 
                   c=color_test, s=3, label = 'Test data')
        ax.scatter(df_temp.target.loc[df_temp.istest==False].values, 
                   df_temp.pred.loc[df_temp.istest==False].values, 
                   c=color_train, s=3, label = 'Train data')
        ax.set_title("Actual vs Predicted in " + daypart)
    # all
    ax = axs[2,1]
    ax.plot([0,5], [0,5], c='k')
    for season in df_reg_filt.season.unique():
        df_temp = df_reg_filt.loc[df_reg_filt.season==season,:]
        ax.scatter(df_temp.target.values, df_temp.pred.values, 
                   c=color_dict[season], s=3, label=season)
        ax.set_title("Actual vs Predicted - seasons" )
    # label all
    for i in np.arange(3):
        for j in np.arange(2):
            ax = axs[i][j]
            ax.set_xlabel('Actual')
            ax.set_ylabel("Predicted")
            ax.set_xlim([0,5])
            ax.set_ylim([0,5])
            ax.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()
    


# MEASURES
def error_measures(model, X, y, verbose=False):
    # split train test
    X_test = X.iloc[model.test_inds_, :]
    y_test = y[model.test_inds_,]
    X_train = X.iloc[model.train_inds_, :]
    y_train = y[model.train_inds_]

    n = len(y_train)
    p = len(X_train.columns)
    p2 = len(X_test.columns)
    if not p==p2:
        print('error')
    
    # Make predictions using the testing set
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    # MAE
    MAE_train = mean_absolute_error(y_train, y_pred_train)
    MAE_test  = mean_absolute_error(y_test, y_pred)
    # MSE
    MSE_train = mean_squared_error(y_train, y_pred_train)
    MSE_test  = mean_squared_error(y_test, y_pred)
    # explained var
    EVA_train = explained_variance_score(y_train, y_pred_train)
    EVA_test  = explained_variance_score(y_test, y_pred)
    # The coefficient of determination: 1 is perfect prediction
    R2_train = r2_score(y_train, y_pred_train)
    R2_test = r2_score(y_test, y_pred)
    # adjusted r2
    Adjr2_train = 1-(1-R2_train)*(n-1)/(n-p-1)
    Adjr2_test  = 1-(1-R2_test)*(n-1)/(n-p-1)
    # aic
    AIC_train = -2*math.log(len(y_train)*MSE_train)+2*p
    AIC_test  = -2*math.log(len(y_test)*MSE_test) +2*p
    
    # save to dict
    meas = {'MAE_train':MAE_train, 'MAE_test':MAE_test, 
           'MSE_train':MSE_train, 'MSE_test':MSE_test,
           'EVA_train':EVA_train, 'EVA_test':EVA_test,
           'R2_train':R2_train, 'R2_test':R2_test,
           'Adjr2_train':Adjr2_train, 'Adjr2_test':Adjr2_test,
           'AIC_train':AIC_train, 'AIC_test':AIC_test}
    
    # print
    if verbose:
        print('Mean absolute error: train %.2f, test %.2f' % (MAE_train, MAE_test))
        print('Mean squared error:  train %.2f, test %.2f' % (MSE_train, MSE_test))
        print('Explained Variance Score (best=1): train %.2f, test %.2f' % (EVA_train, EVA_test))
        print('Coefficient of determination (R2): train %.2f, test %.2f' %(R2_train, R2_test))
        print('Adjusted coeff. of determination:  train %.2f, test %.2f' %(Adjr2_train, Adjr2_test))
        print('AIC: train %.2f, test %.2f' %(AIC_train, AIC_test))
    return meas


def monthly_plots(ols_obj_pacf, X_pacf, y_pacf, 
                  ols_obj_rfecv=[], X_rfecv=[], y_rfecv=[], 
                  model_names=['PACF','RFECV'], num_day_per_week=1):
    if ols_obj_rfecv==[]:
        models = [ols_obj_pacf]
    else:
        models = [ols_obj_pacf, ols_obj_rfecv]
    # colors 
    colors=['blue', 'green']
    # fig
    xTickdist = 4
    xTickLabels = np.tile(np.arange(0,48,xTickdist*0.5), 4*num_day_per_week)
    fig, axs = plt.subplots(12,2,figsize=(20,120))
    # iterate 
    for ind, model in enumerate(models):
        #  predict
        if ind==0:
            X=X_pacf
            y=y_pacf
        else:
            X=X_rfecv
            y=y_rfecv
            
        y_pred = model.predict(X)
        x=np.arange(4*48*num_day_per_week)
        # plot
        for month in np.arange(1,13):
            # LEFT
            ax = axs[month-1,0]
            inds = np.arange((month-1)*48*4*num_day_per_week, month*48*4*num_day_per_week)
            if ind==0:
                ax.plot(x, y[inds], 'r--', label="Actual",ms=1,linewidth=1)
            ax.plot(x, y_pred[inds], linestyle='dashed', c=colors[ind], 
                    label="Prediction " + model_names[ind],linewidth=2,ms=2)
            ax.legend(loc='best');
            ax.set_title('month ' + str(month))
            ax.set_xlim([x[0]-10, x[-1]+10])
            # label
            ax.set_ylabel('Electricity Consumption [KWh]') 
            ax.set_xlabel('Sample Number')
            # RIGHT
            ax = axs[month-1,1]
            ax.plot(x, y[inds]-y_pred[inds], label= "Residual " + model_names[ind],
                    marker='o', linestyle='-', ms=3, c='black',linewidth=1.0)
            ax.set_title('month ' + str(month))
            ax.set_xlim([x[0]-10,x[-1]+10])
            ax.set_ylabel('Residual (actual-pred)')
            ax.set_xlabel('Sample Number')
    plt.tight_layout()
    plt.show()
    

#from statsmodels.sandbox.regression.predstd import wls_prediction_std
#prstd, iv_l, iv_u = wls_prediction_std(model)
#fig, ax = plt.subplots(figsize=(20,6))
#x=np.arange(len(y))
#ax.plot(x[0:500], y[0:500], 'o--', label="data",ms=1)
#ax.plot(x[0:500], model.fittedvalues[0:500], 'r--.', label="OLS",linewidth=1,ms=1)
#ax.plot(x[0:500], iv_u[0:500], 'g--',linewidth=1,ms=1)
#ax.plot(x[0:500], iv_l[0:500], 'g--',linewidth=1,ms=1)
#ax.legend(loc='best');