import torch


def compute_dual(alpha, Y, W, Omega, lambda_):
    '''
    
    '''
    total_alpha = 0
    for tt in range(Y.shape(0)):
        total_alpha += torch.mean(-1.0 * alpha[tt] * Y[tt])
    dual_obj = -0.5 * lambda_ * torch.trace(torch.mm(torch.mm(W, Omega), torch.inverse(W)))
    return dual_obj



def compute_primal(X, Y, W, Omega, lambda_):
    '''
    
    '''
    total_loss = 0
    for t in range(X.shape[0]):
        preds = torch.mv(Y[t]*X[t], W[:,t])
        total_loss += torch.mean(torch.max(torch.zeros(preds.shape), 1.0 - preds))
        
    primal_obj = total_loss + 0.5 * lambda_ * torch.trace(torch.mm(torch.mm(W, Omega), torch.inverse(W)))
    return primal_obj


def compute_rmse(X, Y, W, opts):
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