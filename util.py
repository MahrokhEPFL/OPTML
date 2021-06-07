import torch
import numpy as np

def compute_dual(alpha, Y, W, Omega, lambda_):
    '''
    
    '''
    total_alpha = 0
    for tt in range(Y.shape[0]):
        total_alpha += torch.mean(-1.0 * alpha[tt] * Y[tt])
    dual_obj = -0.5 * lambda_ * torch.trace(torch.mm(torch.mm(W, Omega), torch.inverse(W)))
    return dual_obj



def compute_primal_prev(X, Y, W, Omega, lambda_):
    '''
    
    '''
    total_loss = 0
    for t in range(len(X)):
        preds = torch.mv(Y[t]*X[t], W[:,t])
        total_loss += torch.mean(torch.max(torch.zeros(preds.shape), 1.0 - preds))
        
    primal_obj = total_loss + 0.5 * lambda_ * torch.trace(torch.mm(torch.mm(W, Omega), torch.inverse(W)))
    return primal_obj


def compute_rmse_prev(X, Y, W, opts):
    '''
    
    '''
    m = X.shape[0]
    n = Y.shape[1]
    Y_hat = torch.empty((m, n))
    
    for t in range(m):
        if opts['obj'] == 'R':
            Y_hat[t] = torch.mv(X[t], W[:,t])
        else:
            Y_hat[t] = torch.mv(torch.sign(X[t]), W[:,t])
            
    if opts['avg']:
        all_errs = torch.zeros((m))
        for t in range(m):
            if opts['obj'] == 'R':
                all_errs[t] = torch.sqrt(torch.mean((Y[t] - Y_hat[t]).pow(2)))
            else:
                all_errs[t] = torch.mean((Y[t]!=Y_hat[t]).float())
                
        err = torch.mean(all_errs)
        
    else:
        Y = Y.reshape((m*n))
        Y_hat = Y_hat.reshape((m*n))
        if opts['obj']=='R':
            err = torch.sqrt(torch.mean((Y-Y_hat).pow(2)))
        else:
            err = torch.mean((Y!=Y_hat).float())
    return err

def compute_rmse(X, Y, W, opts):
    '''
    Inputs:
    - X: list of length m, each element is a 2D matrix size n*d of regression features
    - Y: list of length m, each element is a 2D matrix size n*1 of regression targets
    - W: array of size d*m, col t is the weights for task t
    - opts: boolean opts['avg'] -> if true, first compute rmse for each task then average
                                      else, first average MSE of tasks, then compute sqrt
    Note: the original implementation supports both classification and regression errors, 
          through opts['obj'], but we only use regression.
    '''
    m = len(X)    # m=number of tasks (in our prublem, num of households)
    Y_hat = []    # empty list of length m
                  
    # predict
    for t in range(m):
        Y_hat.append(np.matmul(X[t], W[:,t])) # Y_hat[t]: mat size (n_t,1) of predictions for task (household) t
        
    # find MSE for each task
    all_errs = np.zeros(m)
    for t in range(m):
        all_errs[t] = np.mean(((Y[t]-Y_hat[t]).flatten())**2)
    
    # combine errors of different tasks
    if opts['avg']:
        # first compute rmse for each task, then average
        err = np.mean(all_errs**0.5)
    else:
        # first average MSE of tasks, then compute sqrt
        err = np.mean(all_errs)**0.5
        
    return err


def compute_primal(X, Y, W, Omega, lambda_):
    '''
    Primal for regression.
    
    Output: 
    MSE of all tasks and all smaples + regularization penalty 
    
    Note: the original implementation is for classification tasks
    Note: not used in the algorithm, only for evaluation purpose. 
    Note: this is not a federated implementation.
    Note: the original implementation might cause an overflow! 
          this code averages the losses and wouldn't overflow. 
    '''
    # calculate total num of samples from all households
    total_samples = 0   
    for t in range(len(X)):
        total_samples = total_samples + X[t].shape[0]
        
    total_loss = 0 # MSE of all smaples and all tasks
    for t in range(len(X)):
        # predict
        preds = np.matmul(X[t], W[:, t])
        # add squared loss for task t divided by the total samples
        total_loss = total_loss + np.sum(((Y[t]-preds).flatten())**2)/total_samples

    # add regularization term
    reg = np.trace(np.matmul(np.matmul(W, Omega), np.transpose(W)))
    primal_obj = total_loss + 0.5 * lambda_ * reg
    return primal_obj

