import sys
import math
import torch
import syft as sy
import numpy as np
import tensorflow as tf
import statsmodels.api as sm
import matplotlib.pyplot as plt

from SMWrapper import SMWrapper
from load_data import get_data_of_a_person
from construct_dataset import construct_dataset

import gpflow
from gpflow import utilities
from gpflow import kernels
from gpflow.utilities import print_summary, set_trainable

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score

class SyNet(sy.Module):
    def __init__(self, torch_ref, in_dim, out_dim):
        super(SyNet, self).__init__(torch_ref=torch_ref)
        self.linear = self.torch_ref.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        return x
    
    def predict(self, X):
        return self(torch.FloatTensor(X))
    

##################################################################################################################
class Household:
    def __init__(self, house_id, block_num):
        # check if block_num is a number
        if type(block_num)==type("a"):
            block_num = int(block_num[6:])
        self.house_id  = house_id
        self.block_num = block_num
        self.X = []
        self.y = []
        
    
    ##############################################################################################################    
    def construct_dataset(self, lags, step_ahead, options, 
                          crop_years=False, run_rfecv=False, verbose=False):
        self.options = options
        # load consumption data
        cons_data = get_data_of_a_person(block=self.block_num, 
            house_id=self.house_id, crop_years=crop_years, verbose=verbose)
        # check
        if len(cons_data.date)==0:
            print('[Error] no data')
            return
        self.cons_data = cons_data
        # construct dataset
        df_reg, X, y, feat_names = construct_dataset(self.cons_data,
                      step_ahead=step_ahead,
                      lags = lags,
                      dayparts = self.options["dayparts"],
                      filt_days = self.options["filt_days"],
                      feat_cols = self.options["feat_cols"],
                      remove_holiday = self.options["remove_holiday"],
                      replacement_method = self.options["replacement_method"])
        # feat selection
        if run_rfecv:
            lag_rfecv, feat_rfecv = rfecv_selection(self.cons_data, X, y, feat_names, 
                                            verbose=False, plot_fig=False)
            # dataset with selected lags and all feature cols
            df, X, y, feat_names = construct_dataset(self.cons_data,
                                              lags=lag_rfecv,
                                              step_ahead=step_ahead,
                                              dayparts = self.options["dayparts"],
                                              filt_days = self.options["filt_days"],
                                              feat_cols = self.options["feat_cols"],
                                              remove_holiday = self.options["remove_holiday"],
                                              replacement_method = self.options["replacement_method"])
        # done
        self.X=X.to_numpy(dtype='float64', copy=True)
        self.y=y.reshape(-1, 1)
        
    
    
    ##############################################################################################################    
    def train_test_split(self, test_size=0.25, **kwargs):
        # total data to use
        N = kwargs.get('N', self.X.shape[0])
        # Make dataset smaller
        self.X_small = self.X[0:N, :]
        self.y_small = self.y[0:N, :]
        # train and test sets
        train_inds = np.arange(round(N*(1-test_size)))
        test_inds = np.arange(round(N*(1-test_size)), N)
        self.X_train = self.X[train_inds, :]
        self.y_train = self.y[train_inds, :]
        self.X_test  = self.X[test_inds, :]
        self.y_test  = self.y[test_inds, :]
        # has data?
        self.has_train_data = len(self.y_train)>0
        self.has_test_data  = len(self.y_test)>0
        # info
        self.info = {'total_samples':N, 'train_samples':len(self.y_train), 
        'test_samples':len(self.y_test), 'num_features':self.X.shape[1]}
        
    
    ##############################################################################################################    
    def fit_personal_model(self, method, iterations=30, lr=0.1, verbose=False):
        '''
        methods: OLS or Adam for lr, gp
        OLS  -> use the Moore-Penrose pseudoinverse to solve the ls problem
        Adam -> train 1-layer NN with Adam optimizer, starting from random weights,
                number of iterations given by 'iterations', and learning rate 'lr'
        use_train_data: between 0 and 1 -> percentage of training data to use in fitting
        '''
        # check if training data is available
        if not self.has_train_data:
            #print('no train data')
            params = {"linear.weight": torch.Tensor([[0]*self.info["num_features"]]),
                      "linear.bias": torch.Tensor([0])}
            self.params = params
            return
        
        # fit
        if method=='OLS':
            self.personal_lr = SMWrapper(sm.OLS).fit(self.X_train, self.y_train)
            self.params = self.personal_lr.fitted_model_.params # params[0] is the intercept
            
        if method=='Adam':
            model = SyNet(torch, in_dim=self.info['num_features'] , out_dim=1)
            optim = torch.optim.Adam(params=model.parameters(),lr=lr)
            # iterate
            if verbose:
                print('[INFO] losses are printed for evaluation but are not used by the operator')
            for i in range(iterations):
                optim.zero_grad()
                # predict
                output = model(torch.FloatTensor(self.X_train))
                # calculate loss
                loss = torch.nn.functional.mse_loss(output, torch.FloatTensor(self.y_train.reshape(-1, 1)))
                
                # print info
                if i % 10 == 0 and verbose:
                    print("Epoch ", i, " train loss", loss.item())
                loss.backward()
                optim.step()
            self.personal_lr = model
            self.params = self.personal_lr.state_dict()
        
        if method=='GP':
            # Dataset needs to be converted to tensor for GPflow to handle it
            data  = (tf.convert_to_tensor(self.X_train, "float64"), 
                     tf.convert_to_tensor(self.y_train, "float64"))

            # Defining the GP
            kernel = gpflow.kernels.SquaredExponential()
            my_gp  = gpflow.models.GPR(data, kernel=kernel) 

            # Picking an optimizer and training the GP through MLE
            opt = gpflow.optimizers.Scipy()
            opt.minimize(my_gp.training_loss, my_gp.trainable_variables, 
                         tol=1e-11, options=dict(maxiter=1000), method='l-bfgs-b')

            # Let's take a look at its hyperparameters (after training)
            #print_summary(my_gp)
            self.personal_gp = my_gp
            
        # LR model + GP for residuals
        if method=='AdamGP':
            # fit LR
            self.fit_personal_model(method='Adam', iterations=iterations, lr=lr, verbose=verbose)
            # calculate residuals
            self.y_pred_lr_train = self.predict(data=self.X_train, model=self.personal_lr, method='Adam')
            res = self.y_train - self.y_pred_lr_train.reshape(-1, 1)
            # fit GP
            data  = (tf.convert_to_tensor(self.X_train, "float64"), 
                     tf.convert_to_tensor(res, "float64"))
            kernel = gpflow.kernels.SquaredExponential()
            my_gp  = gpflow.models.GPR(data, kernel=kernel) 
            opt = gpflow.optimizers.Scipy()
            opt.minimize(my_gp.training_loss, my_gp.trainable_variables, 
                         tol=1e-11, options=dict(maxiter=1000), method='l-bfgs-b')
            self.residual_gp = my_gp
        
     
    ##############################################################################################################   
    def predict(self, data, method, **kwargs):
        if not data.shape[1] == self.info['num_features']:
            print('[ERROR] number of features doea not match the training data')
            return
        if data.shape[0]==0:
            return
        if method=='OLS':
            model = kwargs.get('model', self.personal_lr)
            return model.predict(data)
        if method=='Adam':
            model = kwargs.get('model', self.personal_lr)
            return model(torch.FloatTensor(data)).data.numpy().flatten()
        if method=='GP':
            if 'model' in kwargs:
                model = kwargs.get('model')
            else:
                model = self.personal_gp
            mean, var = model.predict_f(tf.convert_to_tensor(self.X_train, "float64"))
            return mean[:, 0].numpy(), var[:, 0].numpy()
        if method=='AdamGP':
            model_lr = kwargs.get('model_lr', self.personal_lr)
            model_gp = kwargs.get('model_gp', self.residual_gp)
            pred_lr      = self.predict(data=data, method='Adam', model=model_lr)
            pred_gp, var = self.predict(data=data, method='GP',   model=model_gp)
            return pred_lr+pred_gp, var
        
    
    ##############################################################################################################    
    def evaluate_model(self, method, measures=['MSE'], verbose=False, **kwargs):
    # ALWAYS USE FLATTEN BEFORE CALCULATING ERRORS
        # initialize dict 
        res = {'MSE_train':-1, 'MSE_test':-1}
        # errors on train set
        if self.has_train_data:
            # predict
            if method=='OLS' or method=='Adam':
                self.y_pred_lr_train = self.predict(data=self.X_train, method=method, **kwargs)
                y_pred = self.y_pred_lr_train
            if method=='GP' or method=='AdamGP':
                self.y_pred_gp_train, self.var_train = self.predict(data=self.X_train, method=method, **kwargs)
                y_pred = self.y_pred_gp_train
            # calculate error measures
            y = self.y_train.flatten()
            for meas in measures:
                if meas=='MSE':
                    res['MSE_train'] = np.mean((y_pred-y)**2)
                if meas=='MAE':
                    res['MAE_train'] = mean_absolute_error(y, y_pred)
                if meas=='R2':
                    res['R2_train']  = r2_score(y, y_pred)
                if meas=='Adjr2':
                    n=self.info['train_samples']
                    p=self.info['num_features']
                    res['Adjr2_train'] = 1-(1-res['R2_train'])*(n-1)/(n-p-1)
                if meas=='AIC':
                    res['AIC_train'] = -2*math.log(len(y)*res['MSE_train'])+2*self.info['num_features']

        
        # errors on test set
        if self.has_test_data:
            # predict
            if method=='OLS' or method=='Adam':
                self.y_pred_lr_test = self.predict(data=self.X_test, method=method, **kwargs)
                y_pred = self.y_pred_lr_test
            if method=='GP' or method=='AdamGP':
                self.y_pred_gp_test, self.var_test = self.predict(data=self.X_test, method=method, **kwargs)
                y_pred = self.y_pred_gp_test
            # calculate error measures
            y = self.y_test.flatten()
            for meas in measures:
                if meas=='MSE':
                    res['MSE_test'] = np.mean((y_pred-y)**2)
                if meas=='MAE':
                    res['MAE_test'] = mean_absolute_error(y, y_pred)
                if meas=='R2':
                    res['R2_test']  = r2_score(y, y_pred)
                if meas=='Adjr2':
                    n=self.info['train_samples']
                    p=self.info['num_features']
                    res['Adjr2_test'] = 1-(1-res['R2_test'])*(n-1)/(n-p-1)
                if meas=='AIC':
                    res['AIC_test'] = -2*math.log(len(y)*res['MSE_test'])+2*self.info['num_features']
            
        # print
        if verbose:
            for meas in measures:
                if meas=='MAE':
                    print('Mean absolute error: train %.2f, test %.2f' % (res['MAE_train'], res['MAE_test']))
                if meas=='MSE':
                    print('Mean squared error:  train %.2f, test %.2f' % (res['MSE_train'], res['MSE_test']))
                if meas=='R2':
                    print('Coefficient of determination (R2): train %.2f, test %.2f' %(res['R2_train'], res['R2_test']))
                if meas=='Adjr2':
                    print('Adjusted coeff. of determination:  train %.2f, test %.2f' %(res['Adjr2_train'], res['Adjr2_test']))
                if meas=='AIC':
                    print('AIC: train %.2f, test %.2f' %(res['AIC_train'], res['AIC_test']))
        return res

    
    ##############################################################################################################
    def plot_gp(self):
        # Plotting the results (two standard deviations = 95% confidence)
        fig = plt.figure(figsize=(12,6))
        # GP
        mean = np.concatenate((self.y_pred_gp_train, self.y_pred_gp_test))
        var  = np.concatenate((self.var_train, self.var_test))
        xx   = np.arange(self.info['train_samples']+self.info['test_samples'])
        plt.plot(xx, mean, color='#0072BD', lw=2, label = 'GP mean')
        plt.fill_between(xx,
                         mean - 2 * np.sqrt(var),
                         mean + 2 * np.sqrt(var),
                         color='#0072BD',
                         alpha=0.2, label = 'GP conf. bound')
        # data
        plt.plot(xx, np.concatenate((self.y_train, self.y_test)), "o", color='#484848', ms=3.5, label = 'data points')
        # LR
        y_pred = np.concatenate((self.y_pred_lr_train, self.y_pred_lr_test))
        plt.plot(xx, y_pred, color='orange', lw=1, label = 'linear regression')
        plt.vlines(self.info['train_samples']-1, -0.2, ymax=4.2, colors='k', linestyles='dashed', label='test')
        plt.legend()
        #plt.xlim([0,N])
        plt.tight_layout()
        plt.xlabel('sample number'), plt.ylabel('electricity consumption')
        plt.title('Comparing Gaussian process and linear regression with ' + str(self.info['train_samples']) + ' samples')
        plt.show()
        
        
        
        
    
    
    
    