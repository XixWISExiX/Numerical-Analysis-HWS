# Problem 1
# Write a function find maximum(x, y, z) which returns the
# maximum of the three numbers x, y, z. Do not call any built-in functions, instead
# use “if-else” statements. Test the function on the following triples (1, 2, 3),
# (3, 2, 1) and (1, 3, 2). Include the code that you used to test it.
def find_maximum(x, y, z):
    """
    Computs the maximum of 3 numbers x, y, and z.
    :param x: number x
    :param y: number y
    :param z: number z
    :return: returns the maximum of the 3 numbers.
    """
    if x >= y and x >= z: return x
    if y >= x and y >= z: return y
    else: return z

# Problem 2.
# Write a function sum i plus one squared(n, m) which evaluates the sum ∑m i=n(i + 1)2.
# Test the function on the following values n = 0, m = 3 and n = −3, m = 2.
# Include the code that you used to test it.
def sum_i_plus_one_squared(n, m):
    """
    Computes the sum of the equation (i+1)**2 from n to m
    :param n: The starting number
    :param m: The ending number
    :return: The sum of the equation (i+1)**2 from n to m
    """
    total = 0
    for i in range(n,m+1):
        total += pow(i+1,2)
    return total



# Problem 8
import numpy as np
import matplotlib.pyplot as plt

def difference_sin(x, h):
    """
    Computes the value of the simple difference (sin(x+h) - sin(x)) / h
    and compares it with the derivative
    :param x: value at which we are computing the difference
    :param h: size of the step in the difference
    :return difference: the computed value of (sin(x+h) - sin(x)) / h
    :return abs_error: | (sin(x+h) - sin(x)) / h   - cos(x)|
    """
    difference = (np.sin(x + h) - np.sin(x)) / h
    abs_error = np.abs(np.cos(x) - difference)
    return difference, abs_error

def h2_difference_sin(x, h):
    """
    Computes the value of the simple difference (sin(x+h) - sin(x-h)) / 2h
    and compares it with the derivative
    :param x: value at which we are computing the difference
    :param h: size of the step in the difference
    :return difference: the computed value of (sin(x+h) - sin(x-h)) / 2h
    :return abs_error: | (sin(x+h) - sin(x-h)) / 2h   - cos(x)|
    """
    difference = (np.sin(x + h) - np.sin(x - h)) / (2*h)
    abs_error = np.abs(np.cos(x) - difference)
    return difference, abs_error

if __name__ == '__main__':
    print("Problem 1 tests")
    print(find_maximum(1, 2, 3))
    print(find_maximum(3, 2, 1))
    print(find_maximum(1, 3, 2))
    print("")

    # Problem 2 tests
    print("Problem 2 tests")
    print(sum_i_plus_one_squared(0, 3))
    print(sum_i_plus_one_squared(-3, 2))
    print("")

    # Problem 8 tests
    print("Problem 8 tests")
    # TODO THE h2 formula seems to work better, but need to know why

    # Make an array  with powers of (1/4)^n with n ranging from 1 to 60
    h_values = [np.power(0.25, n) for n in np.arange(1, 61)]
    # Initialize arrays for the results
    differences = np.zeros(60)
    abs_errors = np.zeros(60)

    h2_differences = np.zeros(60)
    h2_abs_errors = np.zeros(60)

    # Compute differences for x = 0.5 and the range of h values
    x = 0.5
    for n in np.arange(60):
        differences[n], abs_errors[n] = difference_sin(x, h_values[n])
        h2_differences[n], h2_abs_errors[n] = h2_difference_sin(x, h_values[n])


    # Plot the computed differences
    plt.figure(figsize=(4, 4))
    plt.plot(h_values, differences, color='r')
    plt.plot(h_values, h2_differences, color='b')
    plt.title('Differences')
    plt.ylabel('Difference')
    plt.xlabel('h')
    plt.show()


    # Plot the computed differences on the log scale for h.
    # This plot is more informative.
    plt.figure(figsize=(4, 4))
    plt.plot(h_values, differences, color='r')
    plt.plot(h_values, h2_differences, color='b')
    plt.xscale("log")
    plt.title('Differences')
    plt.ylabel('Difference')
    plt.xlabel('h [log Scale]')
    plt.show()


    # Plot the absolute errors of approximating the derivative by the differences
    plt.figure(figsize=(4, 4))
    plt.plot(h_values, abs_errors, color='r')
    plt.plot(h_values, h2_abs_errors, color='b')
    plt.xscale("log")
    plt.title('Absolute error')
    plt.ylabel('Absolute error')
    plt.xlabel('h [log Scale]')
    plt.show()

