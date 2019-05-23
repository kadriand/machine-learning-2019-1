import numpy as np


def variance(kernel, points_x, vector_w):
    """
    :param kernel: k(x,y)
    :param points_x: x
    :param vector_w: w
    :return:
    """
    sum_i = 0
    n = len(points_x)
    for x_i in points_x:
        k_i = kernel(vector_w, x_i)
        sum_j = sum(kernel(vector_w, x_j) for x_j in points_x)
        sum_i += np.power(k_i - 1 / n * sum_j, 2)
    return 1 / n / kernel(vector_w, vector_w) * sum_i


def kernel_i(x_1, x_2):
    return np.dot(x_1, x_2)


def kernel_ii(x_1, x_2):
    return np.power(np.dot(x_1, x_2), 2)


def kernel_iii(x_1, x_2):
    return np.power(np.dot(x_1, x_2) + 1, 5)


def kernel_iv(x, z, sigma=1):
    x_minus_z = np.subtract(x, z)
    return np.exp(-np.dot(x_minus_z, x_minus_z) / 2 / sigma ** 2)


def point_c():
    x = [[0, 1], [-1, 3], [2, 4], [3, -1], [-1, -2]]
    w = [[1, 1], [-1, 1]]

    print("w = (1,1)")
    variance_i = variance(kernel_i, x, w[0])
    variance_ii = variance(kernel_ii, x, w[0])
    variance_iii = variance(kernel_iii, x, w[0])
    variance_iv = variance(kernel_iv, x, w[0])

    print(f"i. {variance_i}")
    print(f"ii. {variance_ii}")
    print(f"iii. {variance_iii}")
    print(f"iv. {variance_iv}")

    print("\nw = (-1,1)")
    variance_i = variance(kernel_i, x, w[1])
    variance_ii = variance(kernel_ii, x, w[1])
    variance_iii = variance(kernel_iii, x, w[1])
    variance_iv = variance(kernel_iv, x, w[1])

    print(f"i. {variance_i}")
    print(f"ii. {variance_ii}")
    print(f"iii. {variance_iii}")
    print(f"iv. {variance_iv}")


point_c()
print('done')
