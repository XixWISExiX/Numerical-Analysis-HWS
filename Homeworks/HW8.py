import numpy as np
import matplotlib.pyplot as plt

# C1 --------------------------------------
def divided_diff(x, y):
    '''
    function to calculate the divided
    differences table
    '''
    n = len(y)
    coef = np.zeros([n, n])
    # the first column is y
    coef[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = \
           (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
            
    return coef

def newton_poly(coef, x_data, x):
    '''
    evaluate the newton polynomial 
    at x
    '''
    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x - x_data[n-k])*p
    return p

# Problem 1
x = np.array([0, 2, 3, 4])
y = np.array([5, 4, -7, 10])
# get the divided difference coef
a_s = divided_diff(x, y)[0, :]

# evaluate on new data points
x_new = np.arange(0, 4.1, .1)
y_new = newton_poly(a_s, x, x_new)

plt.figure(figsize = (12, 8))
plt.plot(x, y, 'bo')
plt.plot(x_new, y_new)
plt.xlabel("x")
plt.ylabel("y")
plt.title('C1 Problem 1')
plt.show()


# Problem 2
x = np.array([1, 2, 3, 4])
y = np.array([2, 1, 6, 47])
# get the divided difference coef
a_s = divided_diff(x, y)[0, :]

# evaluate on new data points
x_new = np.arange(1, 4.1, .1)
y_new = newton_poly(a_s, x, x_new)
px_coef = [5, -27, 45, -21]
y_px = np.polyval(px_coef, x_new)
qx_coef = [1, -5, 8, -5, 3]
y_qx = np.polyval(qx_coef, x_new)

plt.figure(figsize = (12, 8))
plt.plot(x, y, 'bo')
plt.plot(x_new, y_new, 'cyan', label="Lagrange Polynomial")
plt.plot(x_new, y_px, 'r:', label="p(x)")
plt.plot(x_new, y_qx, 'm--', label="q(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title('C1 Problem 2')
plt.show()

# C2 --------------------------------------
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

# x = np.array([0, 1, 1.5, 2, 3, 4, 5]) # Evenly spaced out points seem to work well.
# x = np.array([2.1, 2.2, 2.3, 2.4, 2.5]) # Closly put together points seem to lead to more error than if it was spread out.

interpolation_errors = []
taylor_errors = []
plt.figure(figsize = (12, 8))
for i in range(5):
    if i == 0:
        x = np.array([0, 1]) 
    if i == 1:
        x = np.array([0, 1, 2])
    if i == 2:
        x = np.array([0, 1, 2, 3])
    if i == 3:
        x = np.array([0, 1, 2, 3, 4])
    if i == 4:
        x = np.array([0, 1, 2, 3, 4, 5])

    y_sin = np.sin(x)

    # get the divided difference coef
    a_s = divided_diff(x, y_sin)[0, :]
    x_new = np.arange(0, 5.1, .1)

    y_new = newton_poly(a_s, x, x_new)
    y_taylor = []
    for item in x_new:
        y_taylor.append(taylor_sin(i+2, item))
    y_sin_plot = np.sin(x_new)

    plt.plot(x, y_sin, 'bo')
    plt.plot(x_new, y_new, label=f"Lagrange Polynomial n={i+2}")
    plt.plot(x_new, y_taylor, '-.', label=f"Taylor Series sin(x) term={i}")
    # Only plot the base plot once
    if i == 4:
        plt.plot(x_new, y_sin_plot, 'r:', label="sin(x)")
    interpolation_errors.append(np.average(np.abs(y_new - y_sin_plot)))
    taylor_errors.append(np.average(np.abs(y_new - y_taylor)))
plt.title('C2 Plots')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

plt.figure(figsize = (12, 8))
plt.plot([2, 3, 4, 5, 6], interpolation_errors, label="Interpolation Error")
plt.plot([2, 3, 4, 5, 6], taylor_errors, label="Taylor Series sin(x) Error")
plt.xlabel('Iterations starting with n=2')
plt.ylabel('Error')
plt.title('C2 Error Comparision')
plt.legend()
plt.show()

print('C2')
print("Interpolation Errors:", interpolation_errors)
print("Taylor Series sin(x) Errors:", taylor_errors)

print("The more points (and terms), the more accurate the plot is on average.")
print("It seems that Taylor series has a lower error at lower n values, but then interpolation eventually overtakes it in having a lower error.")
print()



# C3 --------------------------------------
plt.figure(figsize = (12, 8))
for i in range(4):
    x = []
    y = []
    if i == 0:
        n = 11
    if i == 1:
        n = 21
    if i == 2:
        n = 31
    if i == 3:
        n = 41
    # Equally spaced nodes
    x = np.linspace(-5, 5, n)
    for j in range(n):
        y_val = 1 / (x[j] ** 2 + 1)
        y.append(y_val)

    # get the divided difference coef
    a_s = divided_diff(x, y)[0, :]
    x_new = np.arange(-5, 5.1, .1)
    y_new = newton_poly(a_s, x, x_new)

    y_actual = []
    for item in x_new:
        y_actual.append(1 / (item ** 2 + 1))

    plt.plot(x, y, 'o')
    plt.plot(x_new, y_new, label=f"Lagrange Polynomial n={n}")
    # Only plot the base plot once
    if i == 3:
        plt.plot(x_new, y_actual, 'r:', label=f"f(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.title('C3')
plt.legend()
plt.show()

print('C3')
print("I think the plot is actually getting worse as you increase the value of n.")
print("Main reason being that the higher interpolation functions are just looking like they became bugged out at the ends.")
print("The Larger the more n values there are it seems the quicker the function seems to be wrong (values close to 0 being approximated wrong).")
print()

# C4 --------------------------------------
def chebyshev_nodes(a, b, n):
    """
    Uses chebyshev Nodes Distribution
    :param a: lower bound range
    :param b: higher bound range
    :param n: number of nodes (n-1)
    :return: the x value of the nodes
    """
    nodes = np.array([
        0.5 * (a + b) + 0.5 * (b - a) * np.cos((2*k + 1) * np.pi / (2 * (n + 1)))
        for k in range(n + 1)
    ])
    return nodes

plt.figure(figsize = (12, 8))
for i in range(4):
    x = []
    y = []
    if i == 0:
        n = 11
    if i == 1:
        n = 21
    if i == 2:
        n = 31
    if i == 3:
        n = 41
    x = chebyshev_nodes(-5, 5, n-1)
    for j in range(n):
        y_val = 1 / (x[j] ** 2 + 1)
        y.append(y_val)

    # get the divided difference coef
    a_s = divided_diff(x, y)[0, :]
    x_new = np.arange(-5, 5.1, .1)
    y_new = newton_poly(a_s, x, x_new)

    y_actual = []
    for item in x_new:
        y_actual.append(1 / (item ** 2 + 1))

    plt.plot(x, y, 'o')
    plt.plot(x_new, y_new, label=f"Lagrange Polynomial n={n}")
    # Only plot the base plot once
    if i == 3:
        plt.plot(x_new, y_actual, 'r:', label=f"f(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.title('C4')
plt.legend()
plt.show()

print('C4')
print("The plot gets better when we increase the value of n & is better than Problem C3.")
print("The main reason would be because the distribution of the nodes via chebyshev's nodes reduces the error terms for large oscillations.")
print()