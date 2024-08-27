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


if __name__ == '__main__':
    # Make an array  with powers of (1/4)^n with n ranging from 1 to 60
    h_values = [np.power(0.25, n) for n in np.arange(1, 61)]
    # Initialize arrays for the results
    differences = np.zeros(60)
    abs_errors = np.zeros(60)

    # Compute differences for x = 0.5 and the range of h values
    x = 0.5
    for n in np.arange(60):
        differences[n], abs_errors[n] = difference_sin(x, h_values[n])


    # Plot the computed differences
    plt.figure(figsize=(4, 4))
    plt.plot(h_values, differences)
    plt.title('Differences')
    plt.ylabel('Difference')
    plt.xlabel('h')
    plt.show()


    # Plot the computed differences on the log scale for h.
    # This plot is more informative.
    plt.figure(figsize=(4, 4))
    plt.plot(h_values, differences)
    plt.xscale("log")
    plt.title('Differences')
    plt.ylabel('Difference')
    plt.xlabel('h [log Scale]')
    plt.show()


    # Plot the absolute errors of approximating the derivative by the differences
    plt.figure(figsize=(4, 4))
    plt.plot(h_values, abs_errors)
    plt.xscale("log")
    plt.title('Absolute error')
    plt.ylabel('Absolute error')
    plt.xlabel('h [log Scale]')
    plt.show()

