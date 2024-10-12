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

def plot_taylor_e(lowest_order, highest_order):
    # Set the points in the interval [0,1]
    n_points = 1000
    x_values = np.linspace(0, 2, n_points)
    y_values = np.zeros(1000)

    # Plot the values computed by the Taylor polynomial and by Numpy
    plt.figure(figsize=(6, 5))

    # Set the order of the polynomial and compute at the given points
    for m in range(lowest_order, highest_order+1):
        for n in np.arange(n_points):
            y_values[n] = taylor_e(m, x_values[n])
        plt.plot(x_values, y_values,linewidth=3, label=("Taylor Polynomial of order %d" % m))
    plt.plot(x_values, np.exp(x_values),linewidth=3, label="Numpy value of the exponential")
    plt.title('Taylor series e^x')
    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

    # Plot the absolute differences between the values computed by the Taylor polynomial and by Numpy
    plt.figure(figsize=(6, 5))
    # Set the order of the polynomial and compute at the given points
    for m in range(lowest_order, highest_order+1):
        for n in np.arange(n_points):
            y_values[n] = taylor_e(m, x_values[n])
        plt.plot(x_values, np.abs(y_values - np.exp(x_values)), linewidth=3, label=("Taylor Polynomial of order %d Difference" % m))
    plt.title('Difference between Taylor series and Numpy e^x')
    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

def print_difference_errors_taylor_e(error):
    n_points = 1000
    x_values = np.linspace(0, 2, n_points)
    y_values = np.zeros(1000)
    for m in range(1, 1000):
        for n in np.arange(n_points):
            y_values[n] = taylor_e(m, x_values[n])
        print(f"Difference in Values at x = 2, m = {m}:", np.abs(y_values[-1] - np.exp(x_values[-1])))
        if np.abs(y_values[-1] - np.exp(x_values[-1])) < error: break;



def taylor_sin(order, x):
    """
    Evaluates the taylor polynomial of sin(x)
    :param order: of the polynomial
    :param x: value at which we approximate sin(x)
    :return: the value of the taylor polynomial
    """

    single_term = x
    result = single_term
    for i in range(1, order + 1):
        single_term = -single_term * x**(2) / ((2*i) * (2*i + 1))
        result += single_term
    return result

def plot_taylor_sin(lowest_order, highest_order):
    # Set the points in the interval [0,1]
    n_points = 1000
    x_values = np.linspace(0, 2, n_points)
    y_values = np.zeros(1000)

    # Plot the values computed by the Taylor polynomial and by Numpy
    plt.figure(figsize=(6, 5))

    # Set the order of the polynomial and compute at the given points
    for m in range(lowest_order, highest_order+1):
        for n in np.arange(n_points):
            y_values[n] = taylor_sin(m, x_values[n])
        plt.plot(x_values, y_values,linewidth=3, label=("Taylor Polynomial of order %d" % m))
    plt.plot(x_values, np.sin(x_values),linewidth=3, label="Numpy value of the sin")
    plt.title('Taylor series sin(x)')
    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

    # Plot the absolute differences between the values computed by the Taylor polynomial and by Numpy
    plt.figure(figsize=(6, 5))
    # Set the order of the polynomial and compute at the given points
    for m in range(lowest_order, highest_order+1):
        for n in np.arange(n_points):
            y_values[n] = taylor_sin(m, x_values[n])
        plt.plot(x_values, np.abs(y_values - np.sin(x_values)), linewidth=3, label=("Taylor Polynomial of order %d Difference" % m))
    plt.title('Difference between Taylor series and Numpy sin(x)')
    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

def print_difference_errors_taylor_sin(error):
    n_points = 1000
    x_values = np.linspace(0, 2, n_points)
    y_values = np.zeros(1000)
    for m in range(1, 1000):
        for n in np.arange(n_points):
            y_values[n] = taylor_sin(m, x_values[n])
        print(f"Difference in Values at x = 2, m = {m}:", np.abs(y_values[-1] - np.sin(x_values[-1])))
        if np.abs(y_values[-1] - np.sin(x_values[-1])) < error: break;


if __name__ == '__main__':

    
    print('---------------\n')
    print('Plot values for order number 3 and absolute errors')
    plot_taylor_e(3, 3)

    print('Q5 Part 1) How do the errors change with increasing order of the polynomial?\n')
    plot_taylor_e(1, 6)

    print('Assuming numpy is our precise value of e^x, the errors seem to decrease with the number of')
    print('increasig order polynomials, so the more terms of the series we compute, the more accurate the answer.')
    print('We can clearly see this with the adjustments I made to the plots showing order 1 polynomials to')
    print('order 6 polynomials getting closer to zero as the number of polynomials in the difference plot increase.')
    print('---------------\n')
    print('Q5 Part 2) What is the smallest order of the polynomial for which all the errors are bounded by 10^(-6)?\n')
    print('I could compute this by hand, but since this is a computer problem, I will solve this problem with my computer.')
    print('Given the range [0,2] the place where there is going to be a higher error separation is at x = 2.')
    print('We can also see this given the plot of differences, at the end where x = 2 being the most dramatic.\n')
    print_difference_errors_taylor_e(10 ** (-6))

    print()
    print('Therefore the order polynomial 13 is the point where all the errors are under 10^(-6)')


    print('===============\n')


    print('Q6) Redo the previous problem for the function f(x) = sin(x).\n')
    print('Plot values for order number 3 and absolute errors')
    plot_taylor_sin(3, 3)

    print('Q6 Part 1) How do the errors change with increasing order of the polynomial?\n')
    plot_taylor_sin(1, 6)

    print('Assuming numpy is our precise value of sin(x), the errors seem to decrease with the number of')
    print('increasig order polynomials, so the more terms of the series we compute, the more accurate the answer.')
    print('We can clearly see this with the adjustments I made to the plots showing order 1 polynomials to')
    print('order 6 polynomials getting closer to zero as the number of polynomials in the difference plot increase.')
    print('---------------\n')
    print('Q5 Part 2) What is the smallest order of the polynomial for which all the errors are bounded by 10^(-6)?\n')
    print('I could compute this by hand, but since this is a computer problem, I will solve this problem with my computer.')
    print('Given the range [0,2] the place where there is going to be a higher error separation is at x = 2.')
    print('We can also see this given the plot of differences, at the end where x = 2 being the most dramatic.\n')
    print_difference_errors_taylor_sin(10 ** (-6))

    print()
    print('Therefore the order polynomial 6 is the point where all the errors are under 10^(-6)')