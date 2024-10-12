import numpy as np
import matplotlib.pyplot as plt


def taylor_e(order, x):
    """
    Evaluates the taylor polynomial of e^x
    :param order: of the polynomial
    :param x: value at which we approximate e^x
    :return: the value of the taylor polynomial
    """

    single_term = 1
    result = single_term
    for i in range(1, order + 1):
        single_term = single_term * x / i
        result += single_term
    return result


if __name__ == '__main__':

    # Evaluate the Taylor polynomial of order 3 on 1000 uniformly spaced points in the interval [0,2]
    # Compare with the exponential function implemented in mumpy

    # Set the points in the interval [0,1]
    n_points = 1000
    x_values = np.linspace(0, 2, n_points)
    y_values = np.zeros(1000)

    # Set the order of the polynomial and compute at the given points
    m = 3
    for n in np.arange(n_points):
        y_values[n] = taylor_e(m, x_values[n])

    # Plot the values computed by the Taylor polynomial and by Numpy
    plt.figure(figsize=(6, 5))
    plt.plot(x_values, y_values,linewidth=3, label=("Taylor Polynomial of order %d" % m))
    plt.plot(x_values, np.exp(x_values),linewidth=3, label="Numpy value of the exponential")
    plt.title('Taylor series')
    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()



    # Plot the absolute differences between the values computed by the Taylor polynomial and by Numpy
    plt.figure(figsize=(6, 5))
    plt.plot(x_values, np.abs(y_values - np.exp(x_values)), linewidth=3)
    plt.title('Difference between Taylor series and Numpy')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()