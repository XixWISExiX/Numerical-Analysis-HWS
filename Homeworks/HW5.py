# Computer Problem Portion of HW5
import numpy as np

# Given bisection algorithm
def bisection(f, a, b, max_iter, tolerance):
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
    for n in np.arange(max_iter):
        error /= 2
        p = a + error
        fp = f(p)

        if fp == 0 or error < tolerance:
            return True, p
        if np.sign(fa) == np.sign(fp):
            a = p
            fa = fp

    return False, 0

########################################################################################
# C1
########################################################################################

import math

# Better bisection algorithm
def better_bisection(f, a, b, max_iter, tolerance):
    """
    Approximates a root p in [a,b] of the function f with error smaller than 
    a given tolerance.
    Note that f(a)f(b) has to be negative for this method to work & the function
    needs to be continuous. There is no continuous check in this algorithm so
    you will need to do that yourself.
    :param f: function whose root is approximated
    :param a: left bound on the position of the root
    :param b: right bond on the position of the root
    :param max_iter: maximum number of iteration to reach the tolerance
    :param tolerance: maximal allowed error

    :return: converged - flag indicating if the method reached the precision
             root - approximation of the root
    """

    # Calculate how long the function will take (find n)
    # 1/(2**n) * abs(b-a) < tolerance
    # abs(b-a) < tolerance * 2**n
    # abs(b-a)/tolerance < 2**n
    # log_2(abs(b-a)/tolerance) = n
    num_of_iter = abs(math.ceil(math.log2(abs(b-a)/tolerance)))

    # If precision cannot be achieved by the algo, don't run it.
    if max_iter < num_of_iter:
        return False, 0


    fa = f(a)
    fb = f(b)

    # If the signs aren't the same, the condition doesn't match to run this algo (saves from underflow)
    if np.sign(fa) == np.sign(fb):
        return False, 0

    error = b - a
    for _ in np.arange(num_of_iter):
        error /= 2
        p = a + error
        fp = f(p)

        if np.sign(fa) == np.sign(fp):
            a = p
            fa = fp

    return True, p


# Test for C1
def f(x):
    '''
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    '''
    return -x + (1/3)

a = 0
b = 1
max_iter = 1000
tolerance = 10 ** (-9)

print("C1 Testing")
print("Bisection algo given",bisection(f, a, b, max_iter, tolerance))
print("C1 Bisection algo", better_bisection(f, a, b, max_iter, tolerance))

########################################################################################
# C2
########################################################################################

import numpy as np

def f(x):
    '''
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    '''
    return math.sqrt(x) - np.cos(x)
    # Could Use Maclaurin Series
    # => x**0.5 - (1 - (x**2)/2 + (x**4)/24)
    # => x**0.5 - 1 + (x**2)/2 - (x**4)/24
    # return math.sqrt(x) - 1 + math.pow(x,2)/2 - math.pow(x,4)/24

a = 0
b = 1
max_iter = 1000
tolerance = 10 ** (-9)

print("\nC2 Results")
print("Bisection algo given",bisection(f, a, b, max_iter, tolerance))
print("C1 Bisection algo", better_bisection(f, a, b, max_iter, tolerance))

########################################################################################
# C3
########################################################################################

def f(x):
    '''
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    '''
    # x**9 + 18*x**3 + 38*x**2 - 57*x - 1
    return x**9 + 18*x**3 + 38*x**2 - 57*x - 1

a = 0
b = 1
max_iter = 1000
tolerance = 10 ** (-9)

print("\nC3 Results")
print("Bisection algo given",bisection(f, a, b, max_iter, tolerance))
print("C1 Bisection algo", better_bisection(f, a, b, max_iter, tolerance))


########################################################################################
# C4
########################################################################################

def f(x):
    '''
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    '''
    # y = x**3 - 2*x + 1 and y = x
    # x = x**3 - 2*x + 1 
    # 0 = x**3 - 3*x + 1 
    return x**3 - 3*x + 1

a = 0
b = 1
max_iter = 1000
tolerance = 10 ** (-9)

print("\nC4 Results")
print("Bisection algo given",bisection(f, a, b, max_iter, tolerance))
print("C1 Bisection algo", better_bisection(f, a, b, max_iter, tolerance))


########################################################################################
# C5
########################################################################################

def f(x):
    '''
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    '''
    return np.tan(x) - x

a = 4
b = 5
max_iter = 1000
tolerance = 10 ** (-9)

print("\nC5 Results")
print("(a) Bisection algo given",bisection(f, a, b, max_iter, tolerance))
print("(a) C1 Bisection algo", better_bisection(f, a, b, max_iter, tolerance))

a = 1
b = 2

print("(b) Bisection algo given",bisection(f, a, b, max_iter, tolerance))
print("(b) C1 Bisection algo", better_bisection(f, a, b, max_iter, tolerance))

print("Explaination: In the equation tan(x) - x, the algorithm gives a root in the interval [1,2].")
print("However, this is actually not the case. The value returned was 1.5707 or pi/2 from the algorithm.")
print("But when you plug that into tan(x) you get undefined.")
print("This is because tan(x) is not a continuous function and therefore violates the other condition")
print("for the algorithm to work, that being the function must be continuous.")

########################################################################################
# C6
########################################################################################

def f(x):
    '''
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    '''
    # x = 3**0.5
    # x**2 = 3
    # x**2 - 3 = 0
    return x**2 - 3

a = 1
b = 2
max_iter = 1000
tolerance = 10 ** (-9)

print("\nC6 Results")
print("Bisection algo given",bisection(f, a, b, max_iter, tolerance))
print("C1 Bisection algo", better_bisection(f, a, b, max_iter, tolerance))

########################################################################################
# C7
########################################################################################

# Algebra
# y = ux + v
# u = rise/run = [f(b)-(fa)]/[b-a]
# so
# y = u(x-a) move to the right by a
# y = u(x-a) + f(a) up by f(a)
# y = {[f(b)-(fa)]/[b-a]}(x-a) + f(a)
# ^ the equation of the secant line
# 0 = {[f(b)-(fa)]/[b-a]}(x-a) + f(a)
# -f(a) = {[f(b)-(fa)]/[b-a]}(x-a)
# -f(a)/{[f(b)-(fa)]/[b-a]} = x-a
# (-f(a)/{[f(b)-(fa)]/[b-a]}) + a = x
# (-f(a)/{[f(b)-(fa)]/[b-a]}) + a = x
# x = (a*fa - b*fa)/(fb-fa) + a
# p1 = x = [a*f(b) - b*f(a)] / [f(b) - f(a)]

def false_position(f, a, b, max_iter, tolerance):
    """
    Approximates a root p in [a,b] of the function f with error smaller than 
    a given tolerance.
    Note that f(a)f(b) has to be negative for this method to work & the function
    needs to be continuous. There is no continuous check in this algorithm so
    you will need to do that yourself.
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

    # If the signs aren't the same, the condition doesn't match to run this algo (saves from underflow)
    if np.sign(fa) == np.sign(fb):
        return False, 0

    p_minus_one = 0 
    for _ in np.arange(max_iter):
        p = (a*fb - b*fa)/(fb-fa)
        fp = f(p)

        if fp == 0 or abs(p-p_minus_one) < tolerance:
            return True, p

        if np.sign(fa) == np.sign(fp):
            a = p
            fa = fp
        else:
            b = p
            fb = fp
        p_minus_one = p

    return False, 0



def f(x):
    '''
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    '''
    return x + 0.9

a = -1
b = 1
max_iter = 1000
tolerance = 10 ** (-9) # 9 decimal places

print("\nC7 Results")
print("(a) False Position Algorithm", false_position(f, a, b, max_iter, tolerance))
print("(a) Bisection Algorithm", better_bisection(f, a, b, max_iter, tolerance))

def f(x):
    '''
    A function f that takes in a singular input x 
    :param x: The input variable for the equation
    :return: The function result 
    '''
    return x**2 - 1.8*x + 0.71

a = -1
b = 1

print("(b) False Position Algorithm", false_position(f, a, b, max_iter, tolerance))
print("(b) Bisection Algorithm", better_bisection(f, a, b, max_iter, tolerance))