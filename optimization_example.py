"""
Created on Mon May 20 12:45:18 2019

P. W. Davenport-Jenkins
University of Manchester
MSc Econometrics
"""

from numba import jit
import scipy.stats as stats
import scipy.optimize as optimize
import numpy as np

# An m vector of moment restrictions
# defined by the user.
@jit
def moment_conditions_matrix(beta, z):
    """
    There are m moment restrictions.
    There are N data points, called z.
    For each data point the m moment restrictions are computed.
    The result is an N x m matrix with each column representing
    a different moment restriction.
    """
    # In this example we do the normal distribution with location 3 and
    # scale 5.
    mu = beta[0]
    sigma = beta[1]
    moments_matrix = np.concatenate(
                           (z - mu,
                           sigma**2-(z - mu)**2,                           
                           (z-mu)**3,
                           3*sigma**4 - (z - mu)**4, 
                           (z-mu)**5,
                           15*sigma**6 - (z - mu)**6),
                            
                        axis=1)       
                    
    return moments_matrix
            

def EL(value):
        return np.log(1 - value)       

def cost_function(x, data):
    
    return -1 * np.mean(EL(data @ x))

normal_random_variable = stats.norm(3,5)

#generate n random values from above distribution
data = normal_random_variable.rvs(size=(1000,1))

beta = [2,4]

"""
Here data_matrix is a 1000 x 6 matrix 
"""
data_matrix = moment_conditions_matrix(beta, data)

"""
Here x is a vector of Lagrange Multipliers and in this example is of length 6.
We require that for each row& in the data_matrix
value = x.T @ row& < 1. This is because we need to compute log(1 - value).

So data_matrix @ x is the vector with rows equal to x.T @ row& for each
row& in the data_matrix.

So with optimize.minimize function in SciPy the inequlity contraint is based
off of Ay - b >= 0 where A is a matrix, y is a vector, and b is a vector.

So in our example we want x.T @ row& < 1 but we don't have strict inequality
availability so we instead, for the sake of trying to make it work, chose to
enforce that x.T @ row& <= 0.5 <===> 0.5 - x.T @ row& >= 0 which in the matrix
form is 0.5 * np.ones(1000) - data_matrix @ x

In the below example, we set this to be the contraint then start the
optimization with the initial choice of lagrange multipliers, x,
being the zero vector.

Then what we want to do is maximise the mean of the vector given by
log(1 - data_matrix @ x). Therefore our cost function is
-1 * np.mean(EL(data @ x)) where EL(value) is defined as np.log(1 - value).

The problem is this doesn't seem to work... I get that the values are
nan's i.e value, in log(1-value), is greater than one, despite the contraint.
Alternatively I get that the optimization simply fails...
"""
cost_function
constraint = [{'type': 'ineq',
               'fun': lambda x: (0.5 * np.ones(1000)) - data_matrix @ x}]

z = optimize.minimize(cost_function,
                      np.zeros(6),
                      args=(data_matrix),
                      method = "SLSQP",
                      constraints=constraint
                      )

