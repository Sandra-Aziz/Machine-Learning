import numpy as np

def  featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    """
    
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X_norm-mu)/sigma
   
    return X_norm



def computeCost(X, y, theta, lamda):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression
    to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    lamda : float
        The regularization parameter.
    
    Returns
    -------
    J : float
        The value of the cost function. 
    """
    m = y.shape[0] # number of training examples
    J = 0
    theta_reg = theta.copy()
    theta_reg[0] = 0
  
    temp = (np.matmul(X,theta.T))-y
    J = ((1/(2*m))* (np.matmul(temp.T,temp)))+(lamda/(2*m)*np.sum(np.square(theta_reg)))
    
    return J


def gradientDescent(X, y, theta, alpha, lamda, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.
        
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    alpha : float
        The learning rate for gradient descent. 
    
    lamda : float
        The regularization parameter.
    
    num_iters : int
        The number of iterations to run gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    """

    m = y.shape[0] # number of training examples
    
    theta_reg = theta.copy()
    theta_reg[0] = 0
    
    J_history = []
    
    for i in range(num_iters):
        
        alphabym=alpha/m
        sumofh0x=np.matmul(X,theta.T)
        theta=theta-((alphabym)*((np.matmul(X.T,sumofh0x-y))+(lamda*theta_reg)))        
        J_history.append(computeCost(X, y, theta,lamda))
    
    return theta, J_history
