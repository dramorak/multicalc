from multicalc import *
import math

e = 2.714
pi = 3.1415

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
    assert fuzzy_equals(_left_limit(id, 0), 0)
    assert fuzzy_equals(_left_limit(square, 2), 4)
    assert fuzzy_equals(_left_limit(sqrt, 4),2)
    assert fuzzy_equals(_left_limit(discrete, 0),0)
    assert fuzzy_equals(_left_limit(discrete, 1),1)
    assert fuzzy_equals(_left_limit(exp,0),1)
    assert fuzzy_equals(_left_limit(sin,0),0)
    assert fuzzy_equals(_left_limit(sin,pi),0)
    assert fuzzy_equals(_left_limit(sin,pi/2),1)
    assert fuzzy_equals(_left_limit(floor,0),-1)

def test__right_limit():
    print("Testing right_limit...")
    assert fuzzy_equals(_right_limit(id, 0), 0)
    assert fuzzy_equals(_right_limit(square, 2), 4)
    assert fuzzy_equals(_right_limit(sqrt, 0),0)
    assert fuzzy_equals(_right_limit(discrete, 0),1)
    assert fuzzy_equals(_right_limit(discrete, 1),1)
    assert fuzzy_equals(_right_limit(exp,0),1)
    assert fuzzy_equals(_right_limit(sin,0),0)
    assert fuzzy_equals(_right_limit(sin,pi),0)
    assert fuzzy_equals(_right_limit(sin,pi/2),1)
    assert fuzzy_equals(_right_limit(floor,0),0)

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
    assert fuzzy_equals(derivative(id, 0), 1)
    assert fuzzy_equals(derivative(square, 2), 4)
    assert fuzzy_equals(derivative(discrete, 1), 0)
    assert fuzzy_equals(derivative(exp, 0), 1)
    assert fuzzy_equals(derivative(exp, 1), e)
    assert fuzzy_equals(derivative(sin, 0), 1)
    assert fuzzy_equals(derivative(sin, pi), -1)
    assert fuzzy_equals(derivative(sin,pi/2), 0)
    

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
    assert function_equals(exp, exp, limit = 10)
    assert function_equals(sin, sin)

    assert function_equals(lin, lin)
    assert function_equals(lin2, lin2)
    assert function_equals(lin3, lin3)
    assert function_equals(quad, quad)
    assert function_equals(multi, multi)

def test__partial_evaluation():
    print("Testing partialEval...")
    f1 = _partial_evaluation(lin, 0, 0)
    f2 = _partial_evaluation(lin2, 0, 0)
    f3 = _partial_evaluation(lin3, 0, 0)
    f4 = _partial_evaluation(quad, 0, 0)
    f5 = _partial_evaluation(multi, 0, 0, 0)

    assert function_equals(f1, lambda x: x)
    assert function_equals(f2, lambda x: x)
    assert function_equals(f3, lambda x: 2*x - 10)
    assert function_equals(f4, lambda x: x**2 + 1)
    assert function_equals(f5, lambda x: 2*x**2 + 10)

def test__partial_derivative():
    print("Testing partial_derivation")
    f1 = partial_derivative(lin, 0)
    f2 = partial_derivative(lin2, 0)
    f3 = partial_derivative(lin3, 0)
    f4 = partial_derivative(quad, 0)
    f5 = partial_derivative(multi, 0)

    assert function_equals(f1, lambda x,y: 1)
    assert function_equals(f2, lambda x,y: 1)
    assert function_equals(f3, lambda x,y: 2)
    assert function_equals(f4, lambda x,y: 2*x + 3*y)
    assert function_equals(f5, lambda x,y,z: y*z + 4*x + 3*y)

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
    assert function_equals(grad4, ans4)
    assert function_equals(grad5, ans5)

def test_all():
    for obj in locals():
        if type(obj) == function:
            incomplete = []
            try:
                obj()
            except:
                incomplete.append(obj)
        print("All tests done.")
        print("The following tests have not yet been implemented:")
        for entry in incomplete:
            print(entry)

if __name__ == "__main__":
    test__left_limit()