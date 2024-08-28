import numpy as np
import matplotlib.pyplot as plt

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
    :return: Returns the maximum of the 3 numbers.
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

# Problem 3
def centigrade_to_fahrenheit(centigrade_arr):
    """
    Computes a Fahrenheit array given a Centigrade array.
    :param centigrade_arr: A numpy array of centigrade values
    :return: The Fahrenheit array.
    """
    return centigrade_arr * (9/5) + 32

# Problem 4
def contains_even_number(arr):
    """
    Computes if the array has a even number.
    :param arr: A numpy array of numbers.
    :return: Whether or not there is an even number in the array.
    """
    for num in arr:
        if num % 2 == 0: return True
    return False

# Problem 5
def number_of_even_numbers(arr):
    """
    Computes if the number of even numbers in a given array.
    :param arr: A numpy array of numbers
    :return: Number of even numbers in the arr array.
    """
    counter = 0
    for num in arr:
        if num % 2 == 0: counter += 1 
    return counter 

# Problem 6
def find_common_values(array1, array2):
    """
    Finds the common values of both given arrays
    :param array1: First numpy array with values
    :param array2: Second numpy array with values
    :return: The common values of both given arrays.
    """
    result = np.array([])
    array1_set = set([])
    for num in array1: array1_set.add(num)
    for num in array2:
        if num in array1_set: result = np.append(result, num)
    return result

# Problem 7
def solve_eqn(a, b, c):
    """
    Computes real solutions for the equation ax^2 + bx + c = 0.
    Using the equation being (-b +- sqrt(b^2 - 4ac)) / 2a.
    :param a: number a
    :param b: number b
    :param c: number c
    :return: The real solutions for the equation ax^2 + bx + c = 0
    """
    if a == 0:
        if b == 0:
            if c == 0:
                return "Infinite solutions"
            else:
                return "No solution"
        else:
            return -c/b

    # Calculates only real solutions
    discriminant = (b**2 - 4*a*c)
    if discriminant > 0:
        root1 = (-b + pow(discriminant, 0.5)) / (2*a)
        root2 = (-b - pow(discriminant, 0.5)) / (2*a)
        return root1, root2
    elif discriminant == 0:
        return -b / (2*a)
    else:
        return "No real solutions"

# Problem 8
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
    print("Problem 1 Tests")
    print(find_maximum(1, 2, 3))
    print(find_maximum(3, 2, 1))
    print(find_maximum(1, 3, 2))
    print("")

    # Problem 2 tests
    print("Problem 2 Tests")
    print(sum_i_plus_one_squared(0, 3))
    print(sum_i_plus_one_squared(-3, 2))
    print("")

    # Problem 3 tests
    print("Problem 3 Tests")
    print(centigrade_to_fahrenheit(np.array([1, 3.4, 75.5, 100.3, -35.2])))
    print("")

    # Problem 4 tests
    print("Problem 4 Tests")
    print(contains_even_number(np.array([1, 3.4, 2, 100.3, 7, 8])))
    print(contains_even_number(np.array([1, 3.4, 3, 100.3, 7, 8.2])))
    print("")

    # Problem 5 tests
    print("Problem 5 Tests")
    print(number_of_even_numbers(np.array([1, 3.4, 2, 100.3, 7, 8])))
    print(number_of_even_numbers(np.array([1, 3.4, 3, 100.3, 7, 8.2])))
    print("")

    # Problem 6 tests
    print("Problem 6 Tests")
    print(find_common_values(np.array([1, 10, 1, 20, 4, 30, 8, 70]), np.array([20, 10, 3, -4, 70, 80])))
    print("")

    # Problem 7 tests
    print("Problem 7 Tests")
    print(solve_eqn(1, -3, 2))
    print(solve_eqn(1, 0, -1))
    print(solve_eqn(1, 0, 1))
    print(solve_eqn(0, -3, 2))
    print("")


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

    print("Problem 8 Analysis")
    print("The h2 or 2h formula seems to be more accurate.")
    print("One reason is that the Absolute error in the initial equation is higher that the latter h2 equation.")
    print("Another reason is that the error term is reduced with the h2 equation, leading to a more accurate answer.")
