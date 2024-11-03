import numpy as np

def trapezoid_method(f, a, b, n):
    '''
    Approximate the integral of f(x) from a to b using the trapezoid method with n intervals.
    
    :param f: The function to integrate.
    :param a: The start of the interval.
    :param b: The end of the interval.
    :param n: The number of intervals.
    :return: The approximate integral of f(x) from a to b.
    '''
    
    h = (b - a) / n
    blocks_sum = 0
    for i in range(n):
        blocks_sum += f(a+h*i) + f(a+h*(i+1))
    return blocks_sum * h/2



b = np.pi
a = 0

# Integral of sin(x) is -cos(x)
print("Actual Value:", -np.cos(b) - (-np.cos(a)))

exponent = -10
n = int(np.round(np.pi / (np.sqrt(10 ** (exponent) * 12/np.pi))))
# NOTE exponent SHOULD be equal to -20, but this is too slow to compute

print("Estimated Value:", trapezoid_method(np.sin, a, b, n))
print("\nNOTE that the exponent variable inside n which is", exponent, "should be -20.")
print("However, the we can still see that the number of accurate places matches the number of the exponent value.")
print("So if exponent is -10 (like it is in this example), then we get 10 places of precision. Hence we can assume that -20 does at least give us 20 places of precision or an error value less than 10^(-20).")

exponent = -20
n = int(np.round(np.pi / (np.sqrt(10 ** (exponent) * 12/np.pi))))
error = 1/12 * (b-a) * 1 * ((b-a)/n)**2 # Error formula for trapizoid method (1 is max for second derivative of sin(x) from 0 to pi)
print("\nError when exponent variable in n is set to ", exponent, ": ", error, sep="")
