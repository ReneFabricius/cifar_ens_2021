import torch
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

class LogisticRegressionTorch:
    def __init__(self, penalty, C=1.0):
        self.coef_ = None
        self.intercept_ = None
        self.penalty_ = penalty
        self.C_ = C
    
    def fit(self, X, y, lr=0.01, epochs=10, optim='lbfgs', verbose=0):
        n, f = X.shape
        y = y.to(dtype=torch.float64)
        self.coef_ = 2 * torch.rand(f, requires_grad=False) - 1
        self.coef_.requires_grad_(True)
        self.intercept_ = torch.zeros(1, requires_grad=True)
        
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
        
        if optim == 'lbfgs':    
            opt = torch.optim.LBFGS(params=(self.coef_, self.intercept_), lr=lr, max_iter=10000)
        elif optim == 'sgd':
            opt = torch.optim.SGD(params=(self.coef_, self.intercept_), lr=lr)
        
        def loss_clos():
            opt.zero_grad()
            lin_comb = torch.sum(X * self.coef_, dim=-1) + self.intercept_
            loss = bce_loss(lin_comb, y)
            if self.penalty_ == "l2":
                l2_pen = torch.sum(self.coef_ * self.coef_)
                loss = loss + l2_pen / self.C_ / 2
            
            loss.backward(retain_graph=True)
            return loss
        
        for e in range(epochs):
            if optim == 'lbfgs':
                loss = opt.step(loss_clos)
            elif optim == 'sgd':
                loss = loss_clos()
                opt.step()
            
            if verbose > 0:
                print("Epoch {}: loss {}".format(e, loss))

        self.coef_.requires_grad_(False)
        self.intercept_.requires_grad_(False)
            
    def predict_proba(self, X):
        n, f = X.shape
        lin_comb = torch.sum(X * self.coef_, dim=-1) + self.intercept_
        pred = torch.special.expit(lin_comb).unsqueeze(1)
        return torch.cat([1 - pred, pred], dim=-1)
    
    def predict_log_proba(self, X):
        return torch.log(self.predict_proba(X))
    
    def score(self, X, y):
        n, f = X.shape
        pred = self.predict_proba(X)
        topv, topi = torch.topk(pred, k=1, dim=1)
        return  torch.sum(topi.squeeze() == y).item() / n
    

class LogisticRegressionTorchPW:
    def __init__(self, penalty='none', C=1.0):
        self.penalty_ = penalty
        self.C_ = C
        self.coef_ = None
        
    def _transform_input(self, X, y):
        n, features = X.shape
        classes = len(torch.unique(y))
        nk = n // classes
        
        X.requires_grad_(False)
        y.requires_grad_(False)
        
        tinds = torch.triu_indices(row=classes, col=classes, offset=1)
        uppr_mask_matrix = torch.zeros(classes, classes, dtype=torch.bool)
        uppr_mask_matrix.index_put_(indices=(tinds[0], tinds[1]), values=torch.tensor([True], dtype=torch.bool))
        uppr_mask = uppr_mask_matrix.unsqueeze(0).expand(2 * nk, classes, classes)
        
        X = torch.permute(X.unsqueeze(2).unsqueeze(3).expand(n, features, classes, classes), (2, 3, 1, 0))
        y = y.unsqueeze(1).unsqueeze(2).expand(n, classes, classes)
        cl = torch.tensor(range(classes)).unsqueeze(1).expand(classes, classes)
        non_diag = torch.eye(classes, classes) != 1
        src_mask = ((y == cl) + (y == cl.t())) * non_diag
        dest_mask = torch.permute(non_diag.unsqueeze(0).expand(2 * nk, classes, classes), (1, 2, 0))
        src_mask = torch.permute(src_mask, (1, 2, 0))
        
        tars_src = torch.zeros(n, classes, classes)
        tars_src[(y == cl) * non_diag] = 1.0
        tars_src = torch.permute(tars_src, (1, 2, 0))
        tars_dest = torch.zeros(classes, classes, 2 * nk)
        tars_dest[dest_mask] = tars_src[src_mask]
        tars_dest = torch.permute(tars_dest, (2, 0, 1))
        
        dest = torch.zeros(classes, classes, features, 2 * nk, dtype=X.dtype)
        src_mask = src_mask.unsqueeze(2).expand(classes, classes, features, n)
        dest_mask = dest_mask.unsqueeze(2).expand(classes, classes, features, 2 * nk)
        dest[dest_mask] = X[src_mask]
        dest = torch.permute(dest, (3, 0, 1, 2))

        return dest, tars_dest, uppr_mask, uppr_mask_matrix

    def fit(self, X, y, max_iter=1000, tolg=1e-5, tolch=1e-9, micro_batch=None, verbose=0):
        n, features = X.shape
        classes = len(torch.unique(y))
        nk = n // classes
        
        dest, tars_dest, uppr_mask, uppr_mask_matrix = self._transform_input(X, y)

        self.coef_ = torch.zeros(size=(classes, classes, features + 1), requires_grad=True, dtype=torch.float64)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
        opt = torch.optim.LBFGS(params=(self.coef_,), lr=1.0, max_iter=max_iter, tolerance_grad=tolg, tolerance_change=tolch)
        
        if micro_batch is None:
            micro_batch = dest.shape[0]
        
        def closure_loss():
            opt.zero_grad()
            Ws = self.coef_[:, :, 0:-1]
            Bs = self.coef_[:, :, -1]
            
            loss = torch.tensor([0], dtype=torch.float64)
            for mbs in range(0, dest.shape[0], micro_batch):
                cur_dest = dest[mbs : mbs + micro_batch]
                cur_tar = tars_dest[mbs : mbs + micro_batch]
                    
                lin_comb = torch.sum(Ws * cur_dest, dim=-1) + Bs
                loss += bce_loss(torch.permute(lin_comb, (1, 2, 0))[uppr_mask_matrix],
                                 torch.permute(cur_tar, (1, 2, 0))[uppr_mask_matrix])
            
            if self.penalty_ == "l2":
                l2_pen = torch.sum(torch.pow(self.coef_[:, :, :-1][uppr_mask_matrix], 2))
                loss += l2_pen / self.C_ / 2
            
            loss.backward(retain_graph=True)
            return loss

        loss = opt.step(closure_loss)
        if verbose > 0:
            print("Loss: {}".format(loss))

        self.coef_.requires_grad_(False)
        
    def set_coefs_from_matrix(self, models):
        classes = len(models)
        feat = len(models[0][1].coef_[0])
        self.coef_ = torch.zeros(classes, classes, feat + 1, dtype=torch.float64) 
        for fc in range(classes):
            for sc in range(fc + 1, classes):
                self.coef_[fc, sc, :-1] = torch.from_numpy(models[fc][sc].coef_[0])
                self.coef_[fc, sc, -1] = torch.from_numpy(models[fc][sc].intercept_)

    def compute_loss(self, X, y):
        dest, tars_dest, uppr_mask, uppr_mask_matrix = self._transform_input(X, y)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        Ws = self.coef_[:, :, 0:-1]
        Bs = self.coef_[:, :, -1]
        lin_comb = torch.sum(Ws * dest, dim=-1) + Bs
            
        loss = bce_loss(torch.permute(lin_comb, (1, 2, 0))[uppr_mask_matrix], torch.permute(tars_dest, (1, 2, 0))[uppr_mask_matrix])
        loss = torch.sum(loss, dim=-1)
            
        if self.penalty_ == "l2":
            l2_pen = torch.sum(torch.pow(self.coef_[:, :, :-1][uppr_mask_matrix], 2), dim=-1)
            loss += l2_pen / self.C_ / 2
        
        return loss
    
    def compute_loss_per_pair(self, X, y):
        classes = len(torch.unique(y))
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="sum") 
        losses = torch.zeros(classes, classes)
        for fc in range(classes):
            for sc in range(fc + 1, classes):
                cur_X = X[(y == fc) + (y == sc)]
                cur_y = y[(y == fc) + (y == sc)]
                mfc = cur_y == fc
                msc = cur_y == sc
                cur_y[msc] = 0
                cur_y[mfc] = 1
                lin_comb = torch.sum(cur_X * self.coef_[fc, sc, :-1], dim=-1) + self.coef_[fc, sc, -1]
                loss = bce_loss(lin_comb, cur_y)
                if self.penalty_ == "l2":
                    l2_pen = torch.sum(self.coef_[fc, sc, :-1] * self.coef_[fc, sc, :-1])
                    loss += l2_pen / self.C_ / 2
                losses[fc, sc] = loss

        return losses 
        
        
def comp_loss(model, X, y, penalty, C):
    pred = model.predict_proba(X)
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)
        
    y = y.to(torch.float64)
    
    loss = torch.nn.BCELoss(reduction="sum")(pred[:, 1], y)
    if penalty == "l2":
        if not torch.is_tensor(model.coef_):
            cf = torch.from_numpy(model.coef_)
        else:
            cf = model.coef_
        l2_pen = torch.sum(cf * cf)
        loss += l2_pen / C / 2
    
    return loss


def comp_weights(n, f, lr=0.01, epochs=10, penalty='none', optim='lbfgs'):
    torch.manual_seed(42)
    X = torch.randn(n, f)
    y = torch.tensor([int(i // (n // 2)) for i in range(n)])
    
    logreg = LogisticRegression(penalty=penalty)
    logreg.fit(X, y)
    
    logregt = LogisticRegressionTorch(penalty=penalty)
    logregt.fit(X, y, lr=lr, epochs=epochs)
    
    return logreg, logregt, X, y

def comp_weights_iris(lr=0.01, epochs=10, penalty="none", optim='lbfgs', C=1.0, verbose=0):
    iris = load_iris()
    X = torch.from_numpy(iris['data'])
    y = torch.from_numpy(iris['target'])
    X = X[y < 2]
    y = y[y < 2]
    
    logreg = LogisticRegression(penalty=penalty, C=C, verbose=verbose, max_iter=1000)
    logreg.fit(X, y)
    
    logregt = LogisticRegressionTorch(penalty=penalty, C=C)
    logregt.fit(X, y, lr=lr, epochs=epochs, optim=optim, verbose=verbose)
    
    lr_loss = comp_loss(logreg, X, y, penalty=penalty, C=C)
    lrt_loss = comp_loss(logregt, X, y, penalty=penalty, C=C)
    print("Loss sklearn: {}, loss torch: {}".format(lr_loss, lrt_loss))
    
    return logreg, logregt, X, y
   

def comp_weights_iris_multcls(penalty="none", C=1.0, max_iter=1000, tolg=1e-5, tolch=1e-9, micro_batch=None, verbose=0):
    iris = load_iris()
    X = torch.from_numpy(iris['data'])
    y = torch.from_numpy(iris['target']).to(torch.float64)
    
    classes = len(torch.unique(y))

    lrpw = LogisticRegressionTorchPW(penalty=penalty, C=C)
    #return lrpw.fit(X, y)
    lrpw.fit(X, y, max_iter=max_iter, tolg=tolg, tolch=tolch, micro_batch=micro_batch)
    
    lrm = [[None for sc in range(classes)] for fc in range(classes)]
    for fc in range(classes):
        for sc in range(fc + 1, classes):
            cur_X = X[(y == fc) + (y == sc)]
            cur_y = y[(y == fc) + (y == sc)]
            mfc = cur_y == fc
            msc = cur_y == sc
            cur_y[msc] = 0
            cur_y[mfc] = 1
            
            lrm[fc][sc] = LogisticRegression(penalty=penalty, C=C, verbose=verbose)
            lrm[fc][sc].fit(cur_X, cur_y)
    
    lr = LogisticRegressionTorchPW(penalty=penalty, C=C)
    lr.set_coefs_from_matrix(lrm)
    
    return lr, lrpw, X, y