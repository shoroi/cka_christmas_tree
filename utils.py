import torch
from torch import nn
import numpy as np
import sklearn.metrics.pairwise as pw

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
#         print(torch.norm(self.centering(L_X)-self.centering(L_Y)))
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
#         print(torch.norm(L_X-L_Y))
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
    
class nonlinCKA_numpy():
    def __init__(self, kernel, gamma):
        super(nonlinCKA_numpy, self).__init__()
        self.kernel = kernel
        self.gamma = gamma
        
    def compute_hsic(self, K, L):
        n = K.shape[0]
        H = np.eye(n)-1/n*np.ones([n,n])
        return np.trace(np.matmul(np.matmul(K,H),np.matmul(L,H)))
    
    def compute_cka(self, A, B):
        if self.kernel == 'rbf': # RBF kernel
            if isinstance(self.gamma, int) or not self.gamma:
                K = pw.rbf_kernel(A, gamma = self.gamma)
                L = pw.rbf_kernel(B, gamma = self.gamma)
            elif self.gamma == 'median_dist':
                A_median = np.median(pw.euclidean_distances(A))
                B_median = np.median(pw.euclidean_distances(B))
            
                K = pw.rbf_kernel(A, gamma = A_median)
                L = pw.rbf_kernel(B, gamma = B_median)
                
        return self.compute_hsic(K,L)/np.sqrt(self.compute_hsic(K,K)*self.compute_hsic(L,L))

class nonlinCKA_torch(nn.Module):
    def __init__(self, kernel, gamma):
        super(nonlinCKA_torch, self).__init__()
        self.kernel = kernel
        self.gamma = gamma
        
    def compute_hsic(self, K, L):
        cuda = torch.device('cuda')
        n = K.shape[0]
        H = torch.eye(n)-1/n*torch.ones([n,n])
        H = H.to(cuda)
        return torch.trace(torch.matmul(torch.matmul(K,H),torch.matmul(L,H)))
    
    def forward(self, A, B):
        cuda = torch.device('cuda')
        if self.kernel == 'rbf': # RBF kernel
#             if isinstance(self.gamma, int) or not self.gamma:
#                 K = pw.rbf_kernel(A, gamma = self.gamma)
#                 L = pw.rbf_kernel(B, gamma = self.gamma)
            if self.gamma == 'median_dist':
                A_median = np.median(pw.euclidean_distances(A.cpu().numpy()))
                B_median = np.median(pw.euclidean_distances(B.cpu().numpy()))
            
                K = torch.Tensor(pw.rbf_kernel(A.cpu().numpy(), gamma = A_median)).to(cuda)
                L = torch.Tensor(pw.rbf_kernel(B.cpu().numpy(), gamma = B_median)).to(cuda)
                
        return self.compute_hsic(K,L)/torch.sqrt(self.compute_hsic(K,K)*self.compute_hsic(L,L))
    
class rbfCKA(nn.Module):
    def __init__(self, median=1):
        super(rbfCKA, self).__init__()
        self.cuda = torch.device('cuda')
        self.median = median
        
    def compute_hsic(self, K, L):
        n = K.shape[0]
        H = (torch.eye(n)-1/n*torch.ones([n,n])).to(self.cuda)
        return torch.trace(torch.matmul(torch.matmul(K,H),torch.matmul(L,H)))
    
    def forward(self, A, B):
        A_dists = torch.Tensor(pw.euclidean_distances(A.cpu().numpy())).to(self.cuda)
        B_dists = torch.Tensor(pw.euclidean_distances(B.cpu().numpy())).to(self.cuda)
        
        A_median = torch.median(A_dists)
        B_median = torch.median(B_dists)
        
        K = self.torch_rbf(A_dists, A_median*self.median)
        L = self.torch_rbf(B_dists, B_median*self.median)
                
        return self.compute_hsic(K,L)/torch.sqrt(self.compute_hsic(K,K)*self.compute_hsic(L,L))
    
#     def torch_euc(self, A, B):
#         n = A.shape[0]
#         dists = torch.zeros([n,n]).to(self.cuda)
#         for i in range(n):
#             for j in range(i,n):
#                 dists[i,j]= torch.sqrt(torch.sum((A[i,:]-B[j,:])**2))
#                 dists[j,i]=dists[i,j]
#         return dists
    
    def torch_rbf(self, dists, gamma):
        rbf = torch.exp(-dists**2/(2*(gamma**2))).to(self.cuda)
        return rbf
    
class rbfCKA_test(nn.Module):
    def __init__(self, median=1):
        super(rbfCKA_test, self).__init__()
        self.cuda = torch.device('cuda')
        self.median = median
        
    def compute_hsic(self, K, L):
        n = K.shape[0]
        H = (torch.eye(n)-1/n*torch.ones([n,n])).to(self.cuda)
        return torch.trace(torch.matmul(torch.matmul(K,H),torch.matmul(L,H)))
    
    def forward(self, A, B):
        A_dists = torch.Tensor(pw.euclidean_distances(A.cpu().numpy())).to(self.cuda)
        B_dists = torch.Tensor(pw.euclidean_distances(B.cpu().numpy())).to(self.cuda)
        
        A_median = torch.median(A_dists)
        B_median = torch.median(B_dists)
        
        K = self.torch_rbf(A_dists, A_median*self.median)
        L = self.torch_rbf(B_dists, B_median*self.median)
                
#         return self.compute_hsic(K,L)/torch.sqrt(self.compute_hsic(K,K)*self.compute_hsic(L,L))
        return (A_median*self.median).cpu().numpy(), (B_median*self.median).cpu().numpy(), torch.sum(K).cpu().numpy(), torch.sum(L).cpu().numpy()
    
#     def torch_euc(self, A, B):
#         n = A.shape[0]
#         dists = torch.zeros([n,n]).to(self.cuda)
#         for i in range(n):
#             for j in range(i,n):
#                 dists[i,j]= torch.sqrt(torch.sum((A[i,:]-B[j,:])**2))
#                 dists[j,i]=dists[i,j]
#         return dists
    
    def torch_rbf(self, dists, gamma):
        rbf = torch.exp(-dists**2/(2*(gamma**2))).to(self.cuda)
        return rbf
        