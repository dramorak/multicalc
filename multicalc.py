"""
Goals:
    A suite of functions to aid in the numerical derivation of multi variable functions. 

TODO:
    infinite limits
    even more partial evaluation (using dicts to identify variables, maybe?)

    **limits aren't defined carefully enough. There's floating point errors everywhere, not enough precision to find answers.
"""
import numpy as np
from inspect import signature
import random
import math

epsilon = 0.000001 #convergence number

class NonConvergent(ArithmeticError):
    pass

def fuzzy_equals(x, y, epsilon = 0.01):
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

def function_equals(f, g, limits=100, d=None):
    """
    Tests to see if two functions are equal on the limits provided. Randomly samples values.
    @param f :: first function to be compared, f::R**d->R
    @param g :: second function to be compared, g::R**d->R
    @param limits :: <optional> tuple detailing the limits of comparison. 
    """

    if d == None:
        sig_f = signature(f)
        sig_g = signature(g)
        d = max(len(sig_f.parameters), len(sig_g.parameters))
    
    for i in range(20000 // d):
        p = rnd_pt(d, limits)
        if not fuzzy_equals(f(*p), g(*p)):
            return False
    return True

def right_limit(f, l):

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
    while abs(f(l + d1) - f(l + d2)) > epsilon**2:
        d1, d2 = d2, d2 / 2
        if n == 10000:
            raise NonConvergent
            
    return f(l + d2)
    """
    return f(l + epsilon**2)
    """

def left_limit(f, l):
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
    while abs(f(l - d1) - f(l - d2)) > epsilon**2:
        d1, d2 = d2, d2 / 2
        n += 1
        if n >= 10000:
            raise NonConvergent
    return f(l - d2)
    """
    return f(l - epsilon**2)
    """   

def limit(f, l):
    """
    Takes a function and returns the limit as x->l
    @param f :: a continuous function from R to R 
    @param l :: the limit to be approached.
    @return  :: a real number corresponding to the limit of x as x->l of f(x).

    @error NonConvergent :: an error if the limit does not exist.
    """

    left = epsilon
    right = epsilon

    n = 0
    while abs(f(l - left) - f(l + right)) > epsilon:
        left = left / 2
        right = right / 2
        n += 1
        if n > 1000:
            raise NonConvergent
    return (f(l - left) + f(l + right)) / 2

def derivative(f):
    """
    Takes a function and returns its derivative.
    @param f :: a continuous function from R to R 
    @return  :: a continuous function from R to R, derivative of f
    """
    def _h(x):
        return limit(lambda h: (f(x+h) - f(x))/h, 0)
    return _h

def partial_evaluation(f, i, *params):
    """
    Takes a function and returns its partial evaluation on params.
    @param f :: a continuous function from R**d to R
    @param i :: an integer corresponding to the i'th paramater of f
    @param params:: d-1 digits corresponding to the arguments of f with the i'th argument removed.
    @return  :: a continuous function from R to R, representing the function with all arguments except the i'th being evaluated.
    """
    params = list(params)
    params.insert(i, None) #not a value yet.
    def _g(a):
        params[i] = a
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
        #find i'th argument.
        params = list(params)
        ith_arg = params.pop(i)

        #partially evaluate f on every parameter except
        partial_eval = partial_evaluation(f, i, *params)
        deriv = derivative(partial_eval)
        return deriv(ith_arg)
    
    return _g
    
def gradient(f, d = None): # :: (R**d -> R) -> d -> (R**d -> (d,1))
    """
    Takes a function and returns its gradient.
    @param f :: a continuous function from R**d1 to R
    @param d1 :: dimensionality of f (please provide, function signatures are inconsistent)
    @return  :: a continuous function from R**d1 to R**d1, the gradient, represented as a size (d1,1) numpy array.
    """

    if d == None:
        d = len(signature(f).parameters) 

    partial_derivatives = []
    for i in range(d):
        partial_derivatives.append(partial_derivative(f, i))

    def _grad(*x):
        #returns the gradient evaluated at x as a (d,1) numpy array).
        ans = np.zeros((d,1))
        for i in range(d):
            ans[i,0] = partial_derivatives[i](*x)
        return ans

    return _grad

def array_pack(f):
    """
    takes f :: R**d -> R and recasts it as a f' :: (d,1) -> R
    @param f:: R**d -> R, normal function
    @return :: (d,1) -> R, a function which takes an np-array of dimension (d,1)
    """
    def _f(x):
        return f(*x[:,0])
    return _f
def array_unpack(f):
    """
    inverse of array_pack
    @param f:: (d,1) -> R, normal function
    @return :: R**d -> R, a function which takes an np-array of dimension (d,1)
    """
    def _f(*args):
        return f(np.array([args]).T)
    return _f

def nparray_gradient(f, d): # :: ((d,1) -> R) -> d -> ((d,1) -> (d,1))
    """
    same as gradient, but f takes an np array of size (d, 1)
    @param f :: a function which takes a (d, 1) np array to R
    @param d :: the dimension of f's domain. 
    @return  :: the gradient of f, grad :: (d, 1) array -> (d,1) array
    """
    
    _f = array_unpack(f)
    grad = gradient(_f, d) #:: R ** d -> (d,1)
    _g = array_pack(grad)
    return _g

if __name__ == "__main__":
    pass