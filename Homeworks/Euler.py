import numpy as np
import matplotlib.pyplot as plt


def euler_method(f, a, b, alpha, steps):
    """
    Euler method for approximating solution of the
    initial value problem x' = f(x,t) for t in [a,b]
    with x(a) = alpha using n steps
    :param f: right hand side of the ODE
    :param a: initial time
    :param b: final time
    :param alpha: initial value condition
    :param steps: number of approximation steps
    :return: approximation of the solution at the mach points
    """

    approximation = np.zeros(steps + 1)
    approximation[0] = alpha
    h = (b - a) / steps
    t = a
    for i in range(steps):
        approximation[i + 1] = approximation[i] + h * f(approximation[i], t)
        t = a + (i+1) * h
    return approximation


def rhs(x, t):
    # Evaluates the function f(x,t) = x - t^2 + 1
    return x - np.square(t) + 1



if __name__ == '__main__':
    # Solve the initial value problem
    # x' = x - t^2 + 1 on [0, 4] and
    # the initial condition x(0) = 0.5

    # Set the parameters
    a = 0
    b = 2
    alpha = 0.5

    plt.figure(figsize=(6, 5))

    # Find the approximations for different numbers of steps
    for n in [10, 20]:
        # Approximate the solution with a given number of steps
        approx = euler_method(rhs, a, b, alpha, n)
        # Sets the time values and plots the approximation
        tt = np.linspace(a, b, n + 1)
        plt.scatter(tt, approx, label="Euler's method with N  = %d" % n)


    # Plots the piecewise linear approximation of the real solution
    tt = np.linspace(a, b, 200)
    solution = np.square(tt + 1) - 0.5 * np.exp(tt)

    print(np.square(2 + 1) - 0.5 * np.exp(2))
    print(euler_method(rhs, a, b, alpha, 4))

    plt.plot(tt, solution, label=r"Precise solution $x(t) = (t+1)^2 - 0.5e^t$")

    plt.legend()
    plt.ylabel('x(t)')
    plt.xlabel('t')
    plt.title(r"Euler's method for $x' = x - t^2 + 1, \quad x(0) = 0.5$", fontsize='small')
    plt.show()
