import numpy as np
from SMWrapper import SMWrapper
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sttools
from construct_dataset import construct_dataset
from analyze_model import error_measures
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFECV

def tune_num_high_corr(df_1p, max_num_feats, max_num_days, feat_cols, filt_days, replacement_method, repeats=10): 
    # pacf
    pacf_val = sttools.pacf(df_1p.energy, nlags=max_num_days*48)
    sorted_lags = np.argsort(-pacf_val)
    # init
    r2_adj_train = [[0 for x in range(max_num_feats)] for y in range(repeats)]
    r2_adj_test  = [[0 for x in range(max_num_feats)] for y in range(repeats)]
    aic_train = [[0 for x in range(max_num_feats)] for y in range(repeats)]
    aic_test  = [[0 for x in range(max_num_feats)] for y in range(repeats)]
    mse_train = [[0 for x in range(max_num_feats)] for y in range(repeats)]
    mse_test  = [[0 for x in range(max_num_feats)] for y in range(repeats)]
    r2_train = [[0 for x in range(max_num_feats)] for y in range(repeats)]
    r2_test  = [[0 for x in range(max_num_feats)] for y in range(repeats)]
    # grid search
    for rep in np.arange(repeats):
        #print('[Info] repeat: ' + str(rep+1))
        for num_feat in np.arange(1,max_num_feats+1):
            #if num_feat%10==0:
                #print('[Info] number of auto-regressors: ' + str(num_feat))
            # add features
            lags = sorted_lags[1:num_feat+1]
            df_reg, X, y, feat_names = construct_dataset(df_1p,
                                         feat_cols=feat_cols,
                                         filt_days = filt_days,
                                         remove_holiday=True,
                                         lags = lags,
                                         replacement_method=replacement_method)
            # fit
            ols_obj = SMWrapper(sm.OLS).cv_fit(X, y, verbose=False)
            # scores
            meas = error_measures(ols_obj, X=X, y=y, verbose=False)
            r2_adj_train[rep][num_feat-1] = meas['Adjr2_train']
            r2_adj_test[rep][num_feat-1]  = meas['Adjr2_test']
            r2_train[rep][num_feat-1] = meas['R2_train']
            r2_test[rep][num_feat-1]  = meas['R2_test']
            aic_train[rep][num_feat-1] = meas['AIC_train']
            aic_test[rep][num_feat-1]  = meas['AIC_test']
            mse_train[rep][num_feat-1] = meas['MSE_train']
            mse_test[rep][num_feat-1]  = meas['MSE_test']
    # find lowest and highest
    maxr2_adj, best_num_r = max((val, idx) for (idx, val) in enumerate(np.mean(r2_adj_test,axis=0)))
    best_num_r=best_num_r+1
    min_mse, best_num_m = min((val, idx) for (idx, val) in enumerate(np.mean(mse_test,axis=0)))
    best_num_m=best_num_m+1
    # find best
    best = np.argmax((r2_adj_test>=0.95*maxr2_adj) * (mse_test<=1.1*min_mse))
    best=best+1
    # plot
    plt.style.use('fivethirtyeight')
    x = np.arange(1,max_num_feats+1)
    # adj R2, aic
    fig,ax = plt.subplots(figsize=(20,10))
    ax.plot(x, np.mean(r2_adj_test,axis=0), 
            c='blue',label = 'test adjusted R2', linewidth=3, ms=1)
    ax.fill_between(x, np.mean(r2_adj_test,axis=0)-1.96*np.std(r2_adj_test,axis=0),
                    np.mean(r2_adj_test,axis=0)+1.96*np.std(r2_adj_test,axis=0),
                    color='purple', alpha=0.1, label='conf. bound test adj. R2')
    ax.plot(x, np.mean(r2_adj_train,axis=0), 
            'b--',label = 'train adj. R2', linewidth=3, ms=1)
    ax.scatter(best_num_r,maxr2_adj,c='red',label = 'highest test adj. R2', s=100, marker='X')
    ax.set_xlabel('Number of auto-regressors')
    ax.set_ylabel('Adjusted R2 Score')
    ax.set_title('Tuning Number of Auto-Regressors')
    # plot aic
    ax2 = ax.twinx()
    ax2.plot(x, np.mean(mse_test,axis=0), 
             c='orange',label = 'MSE test', linewidth=3, ms=1)
    ax2.plot(x, np.mean(mse_train,axis=0), 
             c='orange', linestyle='--',label = 'MSE train', linewidth=3, ms=1)
    ax2.scatter(best_num_m,min_mse,c='red',label = 'lowest test mse', s=100, marker='X')
    ax2.fill_between(x, 
                     np.mean(mse_test,axis=0)-1.96*np.std(mse_test,axis=0), 
                     np.mean(mse_test,axis=0)+1.96*np.std(mse_test,axis=0),
                    color='yellow', alpha=0.3, label='conf. bound test MSE')
    ax2.set_ylabel('MSE')
    fig.legend()
    plt.tight_layout()
    plt.show()
    print('Results:')
    print("Criterion\t\t\tNum.auto-regressors\t\tAdj. R2 test\t\tAIC test\tMSE test")
    print("-------------------------------------------------------------------------------------------------------------")
    print("%s:\t\t\t%1.0f\t\t\t\t%1.2f\t\t\t%1.2f\t\t%1.2f" % ('max adj. R2', best_num_r, 
                                                               np.mean(r2_adj_test, axis=0)[best_num_r-1], 
                                                               np.mean(aic_test, axis=0)[best_num_r-1], 
                                                               np.mean(mse_test, axis=0)[best_num_r-1]))
    print("%s:\t\t\t%1.0f\t\t\t\t%1.2f\t\t\t%1.2f\t\t%1.2f" % ('min MSE', best_num_m, 
                                                               np.mean(r2_adj_test, axis=0)[best_num_m-1], 
                                                               np.mean(aic_test, axis=0)[best_num_m-1], 
                                                               np.mean(mse_test, axis=0)[best_num_m-1]))
    print("%s:\t\t\t%1.0f\t\t\t\t%1.2f\t\t\t%1.2f\t\t%1.2f" % ('first in range', best, 
                                                               np.mean(r2_adj_test, axis=0)[best-1], 
                                                               np.mean(aic_test, axis=0)[best-1], 
                                                               np.mean(mse_test, axis=0)[best-1]))
    print("-------------------------------------------------------------------------------------------------------------")
    return best, pacf_val


def remove_nonsig(ols_obj_hc, feat_names_hc, df_1p, filt_days, replacement_method):
        if not 'constant' in feat_names_hc:
            feat_names_hc = ['constant']+feat_names_hc
        coef_sig = [feat_names_hc[index] for index,value in enumerate(ols_obj_hc.fitted_model_.pvalues) if value<=0.05]
        print('Significant regressors:')
        print(coef_sig)
        print('Removed regressors:')
        print(np.setdiff1d(feat_names_hc, coef_sig))
        
        lag_significant = np.array([int(x.replace('lag ',''))  for x in coef_sig if x.startswith('lag ')])
        feat_significant = [x for x in coef_sig if not x.startswith('lag ') and not x=='constant']

        df_hc_sig, X_sig, y_sig, feat_names_hc_sig = construct_dataset(df_1p,
                                                 feat_cols=feat_significant,
                                                 filt_days = filt_days,
                                                 remove_holiday=True,
                                                 lags = lag_significant,
                                                 replacement_method=replacement_method)
        # Create linear regression object
        ols_obj_sig = SMWrapper(sm.OLS).cv_fit(X_sig, y_sig, verbose=False)
        return ols_obj_sig, X_sig, feat_names_hc_sig
    
    

# *********** RFECV ***************
def scorer_adj_r2(estimator, X, y):
    n = X.shape[0]
    p = X.shape[1]
    y_pred = estimator.predict(X)
    R2 = r2_score(y, y_pred)
    return 1-(1-R2)*(n-1)/(n-p-1)


def rfecv_selection(df_reg, X, y, feat_names, verbose=True, plot_fig=True):
    # add constant column
    X_ext = sm.add_constant(X)
    feat_names = ['constant'] + feat_names

    # set up rfecv
    min_features_to_select=1
    rfecv = RFECV(estimator=SMWrapper(sm.OLS, fit_intercept=False), step=1, cv=5,
                  scoring=scorer_adj_r2,
                  min_features_to_select=min_features_to_select)
    rfecv.fit(X_ext, y)
    
    # selected features
    feat_rfecv = [feat_names[i] for i, x in enumerate(rfecv.support_) if x]

    # Plot number of features VS. cross-validation scores
    if plot_fig:
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation adj. R2 score")
        plt.plot(range(min_features_to_select,
                       len(rfecv.grid_scores_) + min_features_to_select),
                 rfecv.grid_scores_)
        plt.show()

    # display results
    if verbose:
        print("\nRecurssive Feature Elimination + CV")
        print("Optimal number of features: %d" % len(feat_rfecv))
        print("Selected features:")
        print(feat_rfecv)

    # divide 
    lag_rfecv = np.array([int(x.replace('lag ',''))  for x in feat_rfecv if x.startswith('lag ')])
    feat_col_rfecv = [x for x in feat_rfecv if not x.startswith('lag ') and not x=='constant']
    return lag_rfecv, feat_col_rfecv