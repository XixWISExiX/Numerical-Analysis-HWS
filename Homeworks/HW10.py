import numpy as np
import matplotlib.pyplot as plt


def euler_method(f, a, b, alpha, h):
    """
    Euler method for approximating solution of the
    initial value problem x' = f(x,t) for t in [a,b]
    with x(a) = alpha using n steps

    :param f: right hand side of the ODE
    :param a: initial time
    :param b: final time
    :param alpha: initial value condition
    :param h: size of approximation steps
    :return: approximation of the solution at the mach points
    """

    steps = int((b - a) / h)
    # h = (b - a) / steps
    approximation = np.zeros(steps + 1)
    approximation[0] = alpha
    t = a
    for i in range(steps):
        approximation[i + 1] = approximation[i] + h * f(approximation[i], t)
        t = a + (i+1) * h
    return approximation



def euler_method_error(a, t, h, M, L):
    """
    Euler method error for calculating the error of the
    euler method at time step t.

    :param a: initial time
    :param t: time step
    :param h: size of approximation steps
    :param M: Upper bound error of function (rhs) on the interval from a to b
    :param L: Lipschitz constant
    :return: Upperbound of the error in calculating ODE at time step t
    """
    return (h*M)/(2*L) * (np.exp(L*(t-a))-1)

# M = |x''(t)| max[a,b]
# L = df/dx of x' or rhs d/dx

# C1 --------------------------------------

print("C1)")

def rhsA(x, t):
    # Evaluates the function f(x,t) = 1 + (t-x)^2
    return 1 + np.square(t-x)

print("a)", euler_method(rhsA, a=2, b=3, alpha=1, h=0.05)[-1])

def rhsB(x, t):
    # Evaluates the function f(x,t) = t^(-2)(sin(2t) - 2tx)
    return t ** (-2) * (np.sin(2*t) - 2*t*x)

print("b)", euler_method(rhsB, a=1, b=2, alpha=2, h=0.025)[-1])



# C2 --------------------------------------

print()
print("C2)")
print("a)")
print("x(t) = t + 1/(1-t)")
print("x'(t) = 1 + 1/(1-t)^2")
print("x''(t) = 2/(1-t)^3")
for i in range(20):
    t = 2 + i * 0.05
    print(f"Euler's Method Error at time step {i+1} is", abs(euler_method_error(a=2, t=t, h=0.05, M=(2/(1-t)**3), L=2*(1+t))))

print()
print("b)")
print("x(t) = (4+cos(2) - cos(2t))/(2t^2)")
print("x'(t) = (t*sin(2t) + cos(2t) - 4 - cos(2))/t^3")
print("x''(t) = (2t^2*cos(2t) - 4t*sin(2t) - 3*cos(2t) + 3*cos(2) + 12)/t^4")
for i in range(40):
    t = 1 + i * 0.025
    print(f"Euler's Method Error at time step {i+1} is", abs(euler_method_error(a=1, t=t, h=0.05, M=(2*(t**2)*np.cos(2*t) - 4*t*np.sin(2*t) - 3*np.cos(2*t) + 3*np.cos(2) + 12)/(t**4), L=(-2/t))))



# C3 --------------------------------------

print()
print("C3)")

def rhs(x, t):
    # Evaluates the function f(x,t) = -x + t + 1
    return -x + t + 1
# x(t) = e^(-t) + t

print("a)")
print("Euler's method with h=0.2:", euler_method(rhs, a=0, b=5, alpha=1, h=0.2)[-1])
print("Euler's method with h=0.1:", euler_method(rhs, a=0, b=5, alpha=1, h=0.1)[-1])
print("Euler's method with h=0.05:", euler_method(rhs, a=0, b=5, alpha=1, h=0.05)[-1])

print()
print("b)")
t = 5
print("Error for h=0.2:", abs(np.exp(-t) + t - euler_method(rhs, a=0, b=5, alpha=1, h=0.2)[-1]))
print("Error for h=0.1:", abs(np.exp(-t) + t - euler_method(rhs, a=0, b=5, alpha=1, h=0.1)[-1]))
print("Error for h=0.05:", abs(np.exp(-t) + t - euler_method(rhs, a=0, b=5, alpha=1, h=0.05)[-1]))

print()
print("c)")
# x(t) = e^(-t) + t
# x'(t) = 1 - e^(-t)
# x''(t) = e^(-t)
print("Upperbound Error for h=0.2", abs(euler_method_error(a=0, t=t, h=0.2, M=1, L=-1)))
print("Upperbound Error for h=0.1", abs(euler_method_error(a=0, t=t, h=0.1, M=1, L=-1)))
print("Upperbound Error for h=0.05", abs(euler_method_error(a=0, t=t, h=0.05, M=1, L=-1)))

print()
print("d) (takes like 10 sec to compute)")
L = -1
M = 1
a = 0
d = 10**(-6)
lowest_error = np.inf
best_h = 0
for i in range(1000000):
    h = (i+1)*0.000001
    result = 1/L * ((h*M)/2 + d/h) * (np.exp(L*(t-a))-1) + abs(d) * np.exp(L*(t-a))
    if result < lowest_error:
        lowest_error = result
        best_h = h

print("Best h", best_h)
print("Upperbound Error for h=0.001414 (optimal) when d = 10^(-6):", abs(euler_method_error(a=0, t=t, h=0.001414, M=1, L=-1)))



# C4 --------------------------------------

print()
print("C4)")

C = 0.3
R = 1.4
L = 1.7

def E(t):
    return np.exp(-0.06*np.pi*t) * np.sin(2*t-np.pi)

def E_der_1(t):
    return 0.18849*np.exp(-0.06*np.pi*t) * np.sin(2*t) - 2*np.exp(-0.06*np.pi*t)*np.cos(2*t)

def E_der_2(t):
    return 3.96446*np.exp(-0.06*np.pi*t) * np.sin(2*t) + 0.75398*np.exp(-0.06*np.pi*t)*np.cos(2*t)
    
def current_diff_eq(i, t):
    return C*E_der_2(t) + 1/R*E_der_1(t) + 1/L*E(t)

for j in range(100):
    t = 0.1*j
    print(f"Error for h=0.05 at time step {j+1}:", euler_method(current_diff_eq, a=0, b=t, alpha=0, h=0.05)[-1])