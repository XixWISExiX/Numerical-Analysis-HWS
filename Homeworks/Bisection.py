import numpy as np

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





