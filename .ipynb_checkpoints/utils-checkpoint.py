import torch
from torch import nn
import numpy as np

class LinCKA(nn.Module):
    def __init__(self, n=1000):
        super(LinCKA, self).__init__()
        self.resetK(n)
    
    def resetK(self,n):
        unit = torch.ones([n, n])
        I = torch.eye(n)
        H = I - unit / n
        H = H.cuda()
        self.H = H.cuda()
        self.n = n

    def centering(self, K):
        H = self.H
        return torch.matmul(torch.matmul(H, K), H) 

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        print(torch.norm(self.centering(L_X)-self.centering(L_Y)))
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self,X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2),var1,var2

    def forward(self, X,Y):
        if len(X) != self.n:
            self.resetK(len(X))
        return self.linear_CKA(X,Y)
    
class LinCKA2(nn.Module):
    def __init__(self):
        super(LinCKA2, self).__init__()

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        print(torch.norm(L_X-L_Y))
        return torch.sum(L_X * L_Y)

    def forward(self, X,Y):
        n = len(X)
        
        unit = torch.ones([n, n])
        I = torch.eye(n)
        H = I - unit / n
        H = H.cuda()
        
        X = torch.matmul(H,X)
        Y = torch.matmul(H,Y)
        
        HSIC_XY = self.linear_HSIC(X,Y)
        HSIC_XX = self.linear_HSIC(X,X)
        HSIC_YY = self.linear_HSIC(Y,Y)
        
        return HSIC_XY / (torch.sqrt(HSIC_XX)*torch.sqrt(HSIC_YY))