from scipy import *
from scipy.linalg import norm, pinv
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans as kmeans
from matplotlib import pyplot as plt
from numpy import random
import numpy as np

 
class RBF:
     
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = []
        self.betas = np.array([random.uniform(-1, 1) for i in range(numCenters)]) #one beta per centroid
        self.bias = random.uniform(-1, 1, outdim)
        self.W = random.random((self.numCenters, self.outdim))#W matrix of the coefficients we have to optimize
         
    def _basisfunc(self, c, i, x): #i is the index of the center to which we apply the RBF
        return exp(-self.betas[i] * norm(c-x)**2)
    
    def kmns_centers(self, X): #returns numCenters centroids using kmeans method applied to the input X
        X1 = X
        k= self.numCenters
        kmns = kmeans(n_clusters=k, compute_labels=False,
            n_init=1, max_iter=200)
        if X[0].shape == () :#if the data is 1D, we have to reshape it like this : [[],[],...] in order for kmeans to work
            X1 = X.reshape(-1,1)
        kmns.fit(X1)
        c = kmns.cluster_centers_
        if X[0].shape == () :
            return(c.reshape(len(c),))
        else :
            return(c)
            
    def centers_y(self, X, Y): #returns numCenters centroids using kmeans method applied to the whole trainingset (including the desired output Y)
        X1 = X
        k = self.numCenters
        kmns = kmeans(n_clusters = k, compute_labels=False, n_init=1, max_iter = 200)
        if X[0].shape == ():
            D = np.concatenate((X.reshape(-1,1),Y.reshape(-1,1)), axis = 1)
        elif Y[0].shape == () :
            D = np.concatenate((X,Y.reshape(-1,1)), axis=1)
        else:
            D = np.concatenate((X,Y), axis=1)
        kmns.fit(D)
        cy = kmns.cluster_centers_
        if X[0].shape == ():
            c = cy[:,0].reshape(self.numCenters,)
        else:
            c = cy[:,:X[0].shape[0]]
        return(c)
         
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers): #enumerate(X) returns (0, X[0]),(1,X[1])...
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, ci, x) #RBF applied to the (xi)th input vector and (ci)th centroid
        return G
     
    
    def cost(self, params, X, Y): #les parameters are all put in one single array
        i = self.numCenters*self.outdim
        j = i+self.numCenters
        self.W = params[:i].reshape(self.numCenters, self.outdim)
        self.betas = params[i:j]
        self.bias = params[j:]
        Z = self.test(X)
        cost = norm(Z-Y)**2 #cost using the euclidian distance between the output and desired actions
        
        return(cost)
        
    def grad_cost(self, params, X, Y): # computes the cost's gradient. Can be used as a parameter for BFGS optimization method
        grad = np.array([0 for i in range(len(params))])
        for i in range(len(params)):
            p1 = params
            p0 = params
            p1[i]+=0.01
            p0[i]+= -0.01
            grad[i] = (self.cost(p1, X, Y) - self.cost(p0, X, Y))/0.02
        return(grad)
        
    
    def train(self, X, Y): #training using scipy.optimize
        #self.centers = self.kmns_centers(X)
        self.centers = self.centers_y(X,Y)
        
        #we want to optimize W, bias and betas
        W = np.squeeze(np.asarray(self.W.reshape(-1,1))) #transforms matrix into array
        params = np.concatenate((W, self.betas, self.bias))
        # we put all the parameters in one single array
        
        # res = minimize(self.cost, params, args=(X,Y), method = 'BFGS', jac = self.grad_cost, options ={'disp': True})
        res = minimize(self.cost, params, args=(X,Y), method = 'BFGS', options ={'disp': True})
        
        i = self.numCenters*self.outdim
        j = i+self.numCenters
        self.W = res.x[:i].reshape(self.numCenters, self.outdim)
        self.betas = res.x[i:j]
        self.bias = res.x[j:]
    
    def test(self,X): #test using scipy.optimize
        G = self._calcAct(X)
        Y = dot(G, self.W) + self.bias
        if Y[0].shape == (1,):
            Y = Y.reshape(1,-1)
            Y = Y[0]
        return Y


if __name__ == '__main__':

#tests the RBFNN on an example when the input is 3D (like the outcomes of Poppy's actions) and the output is 13D (like the parameters of Poppy's actions)
    
    n = 100
    x = np.array([[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1,1)] for i in range(n)])
    
    y = np.array([[u[0]**2 + u[1]**2, cos(u[0]**3 + u[1]), 3*u[1]-u[0], u[0], u[1], u[2], u[0]+u[1], u[1]+u[2], u[0]+u[2], u[2]**2, 0, 2*u[1], 1] for u in x])
    y.reshape(1,-1)

#I have made the action function of the outcome, but the inverse can be done

    rbf = RBF(3, 10, 13)
    
    rbf.train(x,y)
    
    #we test the network on one example that is not in the training set
    x = np.array([[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1,1)]])
    y = np.array([[u[0]**2 + u[1]**2, cos(u[0]**3 + u[1]), 3*u[1]-u[0], u[0], u[1], u[2], u[0]+u[1], u[1]+u[2], u[0]+u[2], u[2]**2, 0, 2*u[1], 1] for u in x])
    
    z = rbf.test(x)
    print(norm(y-z)**2)
    
    