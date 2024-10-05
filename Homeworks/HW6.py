# Computer Problem Portion of HW5
import numpy as np

def newtonsMethod(f, df, x, max_iterations, epsilon, delta, see_steps=False):
    """
    The function approximates a root of the function f
    starting from the initial guess x
    :param f: formula for the function
    :param df: formula for the derivative of the function
    :param x: initial guess of a root
    :param max_iterations: maximal numberof iterations to be performed
    :param epsilon: parameter for evaluating the accuracy
    :param delta: lower bound on the value of the derivative
    :return: converge: indicates if the method converged 'close' to a root
    :return: root: approximation of the root
    """
    for n in range(1, max_iterations + 1):
        fx = f(x)
        dfx = df(x)

        if np.abs(dfx) < delta:
            return False, 0
        
        h = fx / dfx
        x = x-h

        if see_steps: print('Step', n, '| x =', x)


        if np.abs(h) < epsilon:
            return True, x

    return False, x

# Given bisection algorithm
def bisection(f, a, b, max_iter, tolerance, see_steps=False):
    """
    Approximates a root p in [a,b] of the function f with error smaller than 
    a given tolerance.
    Note that f(a)f(b) has to be negative for this method to work
    :param f: function whose root is approximated
    :param a: left bound on the position of the root
    :param b: right bond on the position of the root
    :param max_iter: maximum number of iteration to reach the tolerance
    :param tolerance: maximal allowed error

    :return: converged - flag indicating if the method reached the precision
             root - approximation of the root
    """

    fa = f(a)
    fb = f(b)

    if np.sign(fa) == np.sign(fb):
        return False, 0

    error = b - a
    for i in np.arange(max_iter):
        error /= 2
        p = a + error
        fp = f(p)

        if see_steps:
            print('Step',i+1, 'x =',p, '| f(p) =', fp)


        if fp == 0 or error < tolerance:
            return True, p
        if np.sign(fa) == np.sign(fp):
            a = p
            fa = fp

    return False, 0

########################################################################################
# C1
########################################################################################
print('C1')

def f(x):
    """
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    """
    return x**5 - 9*x**4 - x**3 + 17*x**2 - 8*x - 8

def f_prime(x):
    """
    The first derivative of function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The first derivative function result 
    """
    return 5*x**4 - 36*x**3 - 3*x**2 + 34*x - 8

max_iter = 100
tolerance = 10 ** (-9)
delta = 0.000001

print(newtonsMethod(f, f_prime, 0, max_iter, tolerance, delta, see_steps=False))
print()
print("Newtons Method with a starting point of x0 = 0 never converges.")
print("The point p oscillates between 0, -1, and 1 as shown by the printed steps taken.")
print("First f(0) = -8, f'(0) = -8, so 0 - (-8)/(-8) = -1 = x1")
print("Second f(0) = 8, f'(0) = -4, so -1 - (8)/(-4) = 1 = x2")
print("Third f(0) = -8, f'(0) = -8, so 1 - (-8)/(-8) = 0 = x3")
print("This cycle continues in a circular manner as shown and never converges.")



########################################################################################
# C2
########################################################################################

print()
print("===================")
print('C2')
print()

def f(x):
    """
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    """
    return x - np.tan(x)

def f_prime(x):
    """
    The first derivative of function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The first derivative function result 
    """
    return 1 - (1/np.cos(x))**2

x = 99
max_iter = 40
tolerance = 10 ** (-9)

# value = 98
# print("Testing Ranges")
# for i in range(1,200):
#     if i % 10 == 0:
#         print()
#     x = value+(i*0.01)
#     print("Value x =", x, "| f(x) =", (x) - np.tan(x))

print('Best Range is 98 to 98.95 for bisection method and these values where found using the above commented out code')
print('Where when one point goes from positive to negative being the main indecator of when passing through 0.')
print("Best starting point for Newton's Method is 98.96 because that's when the value crosses y=0 near 99 radians")
print()

print("Newton's Method:", newtonsMethod(f, f_prime, 98.95, max_iter, tolerance, delta, see_steps=False))
print()


print("Bisection Method:", bisection(f, 98, 98.95, max_iter, tolerance, see_steps=False))

########################################################################################
# C3
########################################################################################

print()
print("===================")
print('C3')
print()

def f(x):
    """
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    """
    return x**4 + 2*x**3 - 7*x**2 + 3

def f_prime(x):
    """
    The first derivative of function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The first derivative function result 
    """
    return 4*x**3 + 6*x**2 - 14*x

max_iter = 40
tolerance = 10 ** (-6)

# value = 0
# print("Testing Ranges")
# for i in range(1,200):
#     if i % 10 == 0:
#         print()
#     x = value+(i*0.1)
#     print("Value x =", x, "| f(x) =", f(x))

print('The Best two starter numbers are 0.8 and 1.6 because those are the two points which are closest to 0 when transitioning')
print('between two numbers separated by 0.1 from 0 to 20.')
print("To correctly get a precision of five significate figures we can set the tolerance to 10^6 and basically inside the Newton's Method")
print("Function we measure the distance between the current step and the previous step. If that distance is less that 10^6, then we know")
print("that our precision is better than 10^5.")
print()

print("Newton's Method for Root 1:", newtonsMethod(f, f_prime, 0.8, max_iter, tolerance, delta, see_steps=False))
print("Newton's Method for Root 2:", newtonsMethod(f, f_prime, 1.6, max_iter, tolerance, delta, see_steps=False))

########################################################################################
# C4
########################################################################################

print()
print("===================")
print('C4')
print()

def f(x):
    """
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    """
    # return (x-1) ** 9
    # return (x-1) ** 20
    # return (x-1) ** 1/4
    # return (x-1) ** (-1)
    return (x-1) ** 2

def f_prime(x):
    """
    The first derivative of function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The first derivative function result 
    """
    # return 9*(x-1) ** 8
    # return 20*(x-1) ** 19
    # return (1/4)*(x-1) ** (-3/4)
    # return (-1)*(x-1) ** (-2)
    return 2*(x-1) ** 1

max_iter = 1000
tolerance = 10 ** (-6)

print("Newton's Method:", newtonsMethod(f, f_prime, 1.1, max_iter, tolerance, delta, see_steps=False))

print("The sequence produced by Newton's method doesn't converge quadratically to the root r = 1.")
print("We can see this through the iterations going on when we use Newton's Method at x = 1.1")
print("1.1 goes to 1.05 goes to 1.025 etc (this is when m = 2, when m > 0 or higher numbers, this becomes worse).")
print("Which means that the function is getting halfed each time at a constant rate and not at a quadratic rate.")
print("When m < 0 Newton's method has a stroke and cannot converge because f'(p) becomes 0.")

########################################################################################
# C5
########################################################################################

print()
print("===================")
print('C5')
print()

def f(x):
    """
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    """
    return np.cos(x) + 1 - np.exp(-x**2)

def f_prime(x):
    """
    The first derivative of function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The first derivative function result 
    """
    return -1*np.sin(x) + 2*x*np.exp(-x**2)

max_iter = 1000
tolerance = 10 ** (-6)

# value = 0
# print("Testing Ranges")
# for i in range(1,401):
#     if i % 10 == 0:
#         print()
#     x = value+(i*0.01)
#     print("Value x =", x, "| f(x) =", f(x))

print("Looping through values 0 to 4 with each iteration being at 0.01, we can see 3.13 is very close to 0.")
print("Newton's Method root in range [0,4]:", newtonsMethod(f, f_prime, 3.13, max_iter, tolerance, delta, see_steps=False))

print()
print("When x0 = 0, the equation never converges because f'(0) = 0")
print("Newton's Method:", newtonsMethod(f, f_prime, 0, max_iter, tolerance, delta, see_steps=False))
print()
print("When x0 = 1, the equation converges at some point where x_hat = 9")
print("Newton's Method:", newtonsMethod(f, f_prime, 1, max_iter, tolerance, delta, see_steps=False))

########################################################################################
# C6
########################################################################################

print()
print("===================")
print('C6')
print('---')
print('R = 400, m = 4')
print()

def f(x):
    """
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    """
    R = 400
    m = 4
    return x**m-R
    # return 1-R/x**m

def f_prime(x):
    """
    The first derivative of function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The first derivative function result 
    """
    R = 400
    m = 4
    return m*x**(m-1)
    # return m*R*x**(-m-1)

def g(x):
    """
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    """
    R = 400
    m = 4
    # return x**m-R
    return 1-R/x**m

def g_prime(x):
    """
    The first derivative of function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The first derivative function result 
    """
    R = 400
    m = 4
    # return m*x**(m-1)
    return (x*(m*R-x**m+R))/(m*R)

max_iter = 100
tolerance = 10 ** (-9)

print("Newton's Method Formula 1, x0 = 10:", newtonsMethod(f, f_prime, 10, max_iter, tolerance, delta, see_steps=False))
print("Newton's Method Formula 2, x0 = 10:", newtonsMethod(g, g_prime, 10, max_iter, tolerance, delta, see_steps=False))
print('Formula one convergest when x0 = 10, but Formula two results in an divergence')
print()

print("Newton's Method Formula 1, x0 = 1:", newtonsMethod(f, f_prime, 1, max_iter, tolerance, delta, see_steps=False))
print("Newton's Method Formula 2, x0 = 1:", newtonsMethod(g, g_prime, 1, max_iter, tolerance, delta, see_steps=False))
print('Formula one and two both converge when x0 = 1, however they converge to different values.')
print()

print('---')
print('R = 0.5, m = 4')
print()

def f(x):
    """
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    """
    R = 0.5
    m = 4
    return x**m-R
    # return 1-(R/x**m)

def f_prime(x):
    """
    The first derivative of function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The first derivative function result 
    """
    R = 0.5
    m = 4
    return m*x**(m-1)
    # return m*R*x**(-m-1)

def g(x):
    """
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    """
    R = 0.5
    m = 4
    # return x**m-R
    return 1-R/x**m

def g_prime(x):
    """
    The first derivative of function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The first derivative function result 
    """
    R = 0.5
    m = 4
    # return m*x**(m-1)
    return (x*(m*R-x**m+R))/(m*R)



print("Newton's Method Formula 1, x0 = 2:", newtonsMethod(f, f_prime, 2, max_iter, tolerance, delta, see_steps=False))
print("Newton's Method Formula 2, x0 = 2:", newtonsMethod(g, g_prime, 2, max_iter, tolerance, delta, see_steps=False))
print("Formula one results in convergence, Formula two results in divergence")
print()

print("Newton's Method Formula 1, x0 = 1:", newtonsMethod(f, f_prime, 1, max_iter, tolerance, delta, see_steps=False))
print("Newton's Method Formula 2, x0 = 1:", newtonsMethod(g, g_prime, 1, max_iter, tolerance, delta, see_steps=False))
print('Formula one and two both converge when x0 = 1, however they converge to different values.')
print()

print("Newton's Method Formula 1, x0 = 0.1:", newtonsMethod(f, f_prime, 0.1, max_iter, tolerance, delta, see_steps=False))
print("Newton's Method Formula 2, x0 = 0.1:", newtonsMethod(g, g_prime, 0.1, max_iter, tolerance, delta, see_steps=False))
print('Formula one and two both converge when x0 = 1, however they converge to different values.')
print()

########################################################################################
# C7
########################################################################################

print()
print("===================")
print('C7')
print()

def d_prime(x):
    """
    A function d_prime that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    """
    # x**2
    # for point (0,1)
    # (x-0)**2 + (y - 1)^2
    # (x-0)**2 + (x^2 - 1)^2
    # d(x) = (x^2) + (x^2 - 1)^2
    # d(x) = (x^2) + (x^2 - 1) (x^2 - 1)
    # d(x) = (x^2) + (x^4 -2x^2 + 1)
    # d(x) = (x^4 - x^2 + 1)
    # d'(x) = 4x^3 - 2x
    return 4*x**3 - 2*x


def d_double_prime(x):
    """
    The derivative of function d_prime that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The first derivative function result 
    """
    # d'(x) = 4x^3 - 2x
    # d''(x) = 12x^2 - 2
    return 12*x**2 - 2


max_iter = 1000
tolerance = 10 ** (-6)

# value = -2
# print("Testing Ranges")
# for i in range(1,41):
#     if i % 10 == 0:
#         print()
#     x = value+(i*0.1)
#     print("Value x =", x, "| f(x) =", d_prime(x))

print('Math is shown in functions, but to solve the distance with point vs line we need')
print("To find d'(p) = 0, where d is the point vs line equation which we will set up.")
print("This mean we must find the d''(p) in order to find the next point with Newton's method.")
print('x0 was found by finding the d_prime values from -2 to 2 with step distance 0.1, of which 0.7 was')
print('one of these values.')
print("Newton's Method:", newtonsMethod(d_prime, d_double_prime, 0.7, max_iter, tolerance, delta, see_steps=False))