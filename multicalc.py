"""
Goals:
    A suite of functions to aid in the numerical derivation of multi variable functions. 

TODO:
    limits of multivariable functions, not just single variable functions.
    infinite limits
    even more partial evaluation (using dicts to identify variables, maybe?)
"""
import numpy as np
from inspect import signature
import random
import math

epsilon = 0.0001 #convergence number
class NonConvergent(ArithmeticError):
    pass

def fuzzy_equals(x, y):
    """
    returns True if x ~= y.
    @params x,y :: 2 values of the same type
    @returns :: bool, true if x~=y, False otherwise.
    """
    d = x - y

    if type(d) == np.ndarray:
        return np.linalg.norm(d) < epsilon
    else:
        return abs(d) < epsilon
        
def rnd_pt(d, limits=100):
    """
    Magics up a random point of dimension d, bounded by limits.
    @params d :: the dimension of the point
    @params limits:: the bound of any of the individual arguments in the pt.
    @returns :: a python list of length d representing the point.
    """
    pt = []
    for i in range(d):
        pt.append(random.random() * limits)
    
    return pt

def function_equals(f, g, limits=100):
    """
    Tests to see if two functions are equal on the limits provided. Randomly samples values.
    @param f :: first function to be compared, f::R**d->R
    @param g :: second function to be compared, g::R**d->R
    @param limits :: <optional> tuple detailing the limits of comparison. 
    """

    sig_f = signature(f)
    sig_g = signature(g)

    if len(sig_f.parameters) != len(sig_g.parameters):
        return False

    d = len(sig_f.parameters)
    for i in range(20000 // d):
        p = rnd_pt(d, limits)
        if not fuzzy_equals(f(*p), g(*p)):
            return False
    return True

def _right_limit(f, l):
    """
    Takes a function and takes the right limit as x -> l 
    @param f :: a continuous function from R to R 
    @param l :: the limit to be approached.
    @return  :: a real number corresponding to the right limit of x as x->l of f(x)

    @error NonConvergent :: an error if the limit does not exist.
    """

    d1 = 1
    d2 = 0.5
    n = 0
    while abs(f(l + d1) - f(l + d2)) > epsilon:
        d1, d2 = d2, d2 / 2
        if n == 10000:
            raise NonConvergent
    return f(l + d2)

def _left_limit(f, l):
    """
    Takes a function and takes the left limit as x -> l
    @param f :: a continuous function from R to R 
    @param l :: the limit to be approached.
    @return  :: a real number corresponding to the left limit of x as x->l of f(x)
    
    @error NonConvergent :: an error if the limit does not exist.
    """

    d1 = 1
    d2 = 0.5
    n = 0
    while abs(f(l - d1) - f(l - d2)) > epsilon:
        d1, d2 = d2, d2 / 2
        n += 1
        if n >= 10000:
            raise NonConvergent
    return f(l - d2)

def limit(f, l):
    """
    Takes a function and returns the limit as x->l
    @param f :: a continuous function from R to R 
    @param l :: the limit to be approached.
    @return  :: a real number corresponding to the limit of x as x->l of f(x).

    @error NonConvergent :: an error if the limit does not exist.
    """

    left = _left_limit(f, l)
    right = _right_limit(f, l)

    if left != right:
        raise NonConvergent

    return (left + right) / 2

def derivative(f):
    """
    Takes a function and returns its derivative.
    @param f :: a continuous function from R to R 
    @return  :: a continuous function from R to R, derivative of f
    """
    def _h(x):
        return limit(lambda h: (f(x+h) - f(x))/h, 0)
    return _h

def _partial_evaluation(f, i, *params):
    """
    Takes a function and returns its partial evaluation on params.
    @param f :: a continuous function from R**d to R
    @param i :: an integer corresponding to the i'th paramater of f
    @param params:: d-1 digits corresponding to the arguments of f with the i'th argument removed.
    @return  :: a continuous function from R to R, representing the function with all arguments except the i'th being evaluated.
    """

    def _g(a):
        params.insert(i, a)
        return f(*params)
    return _g

def partial_derivative(f, i):
    """
    Takes a function and returns its partial derivative wrt the i'th parameter.
    @param f :: a continuous function from R**d to R
    @param i :: an integer corresponding to the i'th paramater of f
    @return  :: a continuous function from R**d to R, the partial derivative of f with respect to the i'th paramater
    """
    
    def _g(*params):
        ith_arg = params.pop(i)
        partial_eval = _partial_evaluation(f, i, params)
        deriv = derivative(partial_eval)
        return deriv(ith_arg)
    
    return _g
        


def gradient(f):
    """
    Takes a function and returns its gradient.
    @param f :: a continuous function from R**d1 to R
    @return  :: a continuous function from R**d1 to R**d1, the gradient, represented as a size (d1,1) numpy array.
    """

    d1 = len(signature(f).parameters)

    partial_derivatives = []
    for i in range(d1):
        partial_derivatives.append(partial_derivative(f, i))

    def _grad(x):
        #returns the gradient evaluated at x as a (d,1) numpy array).
        ans = np.zeros((d1,1))
        for i in range(d1):
            ans[i,0] = partial_derivatives[i](x)
        return ans

    return _grad