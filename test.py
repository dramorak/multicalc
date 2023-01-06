from multicalc import *
import math
e = 2.718281
pi = 3.1415926

id = lambda x: x
square = lambda x: x**2
sqrt = lambda x: x ** (0.5)
def discrete(x):
    if x < 0:
        return 0
    else:
        return 1
exp = lambda x: e ** x
sin = math.sin
floor = math.floor

def test__left_limit():
    print("Testing left_limit...")
    assert fuzzy_equals(left_limit(id, 0), 0)
    assert fuzzy_equals(left_limit(square, 2), 4)
    assert fuzzy_equals(left_limit(sqrt, 4),2)
    assert fuzzy_equals(left_limit(discrete, 0),0)
    assert fuzzy_equals(left_limit(discrete, 1),1)
    assert fuzzy_equals(left_limit(exp,0),1)
    assert fuzzy_equals(left_limit(sin,0),0)
    assert fuzzy_equals(left_limit(sin,pi),0)
    assert fuzzy_equals(left_limit(sin,pi/2),1)
    assert fuzzy_equals(left_limit(floor,0),-1)

def test__right_limit():
    print("Testing right_limit...")
    assert fuzzy_equals(right_limit(id, 0), 0)
    assert fuzzy_equals(right_limit(square, 2), 4)
    assert fuzzy_equals(right_limit(sqrt, 0),0)
    assert fuzzy_equals(right_limit(discrete, 0),1)
    assert fuzzy_equals(right_limit(discrete, 1),1)
    assert fuzzy_equals(right_limit(exp,0),1)
    assert fuzzy_equals(right_limit(sin,0),0)
    assert fuzzy_equals(right_limit(sin,pi),0)
    assert fuzzy_equals(right_limit(sin,pi/2),1)
    assert fuzzy_equals(right_limit(floor,0),0)

def test__limit():
    print("Testing limit...")
    assert fuzzy_equals(limit(id, 0), 0)
    assert fuzzy_equals(limit(square, 2), 4)
    assert fuzzy_equals(limit(discrete, 1),1)
    assert fuzzy_equals(limit(exp,0),1)
    assert fuzzy_equals(limit(sin,0),0)
    assert fuzzy_equals(limit(sin,pi),0)
    assert fuzzy_equals(limit(sin,pi/2),1)

    #sqrt non existence.
    #discrete non existence.
    #floor non existence.
    try:
        limit(discrete, 0)
        limit(floor, 0)
    except NonConvergent:
        pass

def test__derivative():
    print("Testing derivative...")
    assert fuzzy_equals(derivative(id)(0), 1)
    assert fuzzy_equals(derivative(square)(2), 4)
    assert fuzzy_equals(derivative(discrete)(1), 0)
    assert fuzzy_equals(derivative(exp)(0), 1)
    assert fuzzy_equals(derivative(exp)(1), e)
    assert fuzzy_equals(derivative(sin)(0), 1)
    assert fuzzy_equals(derivative(sin)(pi), -1)
    assert fuzzy_equals(derivative(sin)(pi/2), 0)
    

#multivariable functions
lin = lambda x, y: x + y
lin2 = lambda x,y: x - y
lin3 = lambda x,y: 2*x - 3*y + 10
quad = lambda x,y: x**2 + 3*x*y + y**2 + 1
multi = lambda x,y,z: x*y*z + 2*x**2 + 3*x*y - y*z + 10
def test__function_equals():
    print("Testing function_equals...")
    assert function_equals(id, id)
    assert function_equals(square, square)
    assert function_equals(discrete, discrete)
    assert function_equals(exp, exp, limits = 10)
    assert function_equals(sin, sin)

    assert function_equals(lin, lin)
    assert function_equals(lin2, lin2)
    assert function_equals(lin3, lin3)
    assert function_equals(quad, quad)
    assert function_equals(multi, multi)

    assert function_equals(lambda x,y: np.array([[x,y]]), lambda x,y: np.array([[x,y]]))

def test__partial_evaluation():
    print("Testing partialEval...")
    f1 = partial_evaluation(lin, 0, 0)
    f2 = partial_evaluation(lin2, 0, 0)
    f3 = partial_evaluation(lin3, 0, 0)
    f4 = partial_evaluation(quad, 0, 0)
    f5 = partial_evaluation(multi, 0, 0, 0)

    assert function_equals(f1, lambda x: x)
    assert function_equals(f2, lambda x: x)
    assert function_equals(f3, lambda x: 2*x + 10)
    assert function_equals(f4, lambda x: x**2 + 1)
    assert function_equals(f5, lambda x: 2*x**2 + 10)
    
def test__partial_derivative():
    print("Testing partial_derivation...")
    f1 = partial_derivative(lin, 0)
    f2 = partial_derivative(lin2, 0)
    f3 = partial_derivative(lin3, 0)
    f4 = partial_derivative(quad, 0)
    f5 = partial_derivative(multi, 0)

    assert function_equals(f1, lambda x,y: 1)
    assert function_equals(f2, lambda x,y: 1)
    assert function_equals(f3, lambda x,y: 2)
    #assert function_equals(f4, lambda x,y: 2*x + 3*y)
    #assert function_equals(f5, lambda x,y,z: 4*x + y * (z + 3))

def test__gradient(): 
    print("Testing gradient...")
    #simple sanity checks first
    grad_1d_1 = gradient(id)
    grad_1d_2 = gradient(square)
    grad_1d_3 = gradient(exp)
    grad_1d_4 = gradient(sin)
    assert function_equals(lambda x: grad_1d_1(x)[0,0], derivative(id))
    assert function_equals(lambda x: grad_1d_2(x)[0,0], derivative(square))
    assert function_equals(lambda x: grad_1d_3(x)[0,0], derivative(exp))
    assert function_equals(lambda x: grad_1d_4(x)[0,0], derivative(sin))
    #multivariable gradients
    grad1 = gradient(lin)
    grad2 = gradient(lin2)
    grad3 = gradient(lin3)
    grad4 = gradient(quad)
    grad5 = gradient(multi)

    ans1 = lambda x,y: np.array([[1, 1]]).T
    ans2 = lambda x,y: np.array([[1, -1]]).T
    ans3 = lambda x,y: np.array([[2, -3]]).T
    ans4 = lambda x,y: np.array([[2*x + 3*y, 3*x + 2*y]]).T
    ans5 = lambda x,y,z: np.array([[y*z + 4*x + 3*y, x*z + 3*z - z, x*y - y]]).T

    assert function_equals(grad1, ans1)
    assert function_equals(grad2, ans2)
    assert function_equals(grad3, ans3)
    #assert function_equals(grad4, ans4)
    #assert function_equals(grad5, ans5)

nparray_lin = lambda x: lin(*x[:,0]) 
nparray_lin2 = lambda x: lin2(*x[:,0])
nparray_lin3 = lambda x: lin3(*x[:,0])
nparray_quad = lambda x: quad(*x[:,0])
nparray_multi = lambda x: multi(*x[:,0])

"""
NOTE
lin = lambda x, y: x + y
lin2 = lambda x,y: x - y
lin3 = lambda x,y: 2*x - 3*y + 10
quad = lambda x,y: x**2 + 3*x*y + y**2 + 1
multi = lambda x,y,z: x*y*z + 2*x**2 + 3*x*y - y*z + 10
"""

def _test_fcns():
    i2 = np.array([[1,1]]).T
    i3 = np.array([[1,1,1]]).T

    assert nparray_lin(i2) == 2
    assert nparray_lin2(i2) == 0
    assert nparray_lin3(i2) == 9
    assert nparray_quad(i2) == 6
    assert nparray_multi(i3) == 15

def test__nparray_gradient():
    print("Testing nparray_gradient...")
    grad1 = nparray_gradient(nparray_lin, 2)
    grad2 = nparray_gradient(nparray_lin2, 2)
    grad3 = nparray_gradient(nparray_lin3, 2)
    grad4 = nparray_gradient(nparray_quad, 2)
    grad5 = nparray_gradient(nparray_multi, 3)

    ans1 = lambda x: np.array([[1, 1]]).T
    ans2 = lambda x: np.array([[1, -1]]).T
    ans3 = lambda x: np.array([[2, -3]]).T
    ans4 = lambda x: np.array([[2*x[0,0] + 3*x[1,0], 3*x[0,0] + 2*[1,0]]]).T
    ans5 = lambda x: np.array([[x[1,0]*x[2,0] + 4*x[0,0] + 3*x[0,0], x[0,0]*z[2,0] + 3*x[2,0] - x[2,0], x[0,0]*x[2,0] - x[1,0]]]).T

    assert function_equals(array_unpack(grad1), array_unpack(ans1), d=2)
    assert function_equals(array_unpack(grad2), array_unpack(ans2), d=2)
    assert function_equals(array_unpack(grad2), array_unpack(ans2), d=2)

def test_all():
    print("Running complete diagnostic...")

    test__left_limit()
    test__right_limit()
    test__limit()
    test__derivative()
    test__function_equals()
    test__partial_evaluation()
    test__partial_derivative()
    test__gradient()
    _test_fcns()
    test__nparray_gradient()

    print("\n\n Test done.")
if __name__ == "__main__":
    test__nparray_gradient()
    
    
