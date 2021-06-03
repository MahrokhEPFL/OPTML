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


def compute_rmse()