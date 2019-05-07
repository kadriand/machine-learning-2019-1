import numpy as np
from sklearn import datasets
from assignment2.commons import plot_classes, assignment_header
import scipy.stats as stats

iris = datasets.load_iris()
y = iris.target
means = np.zeros((3, 2))
covariances = np.zeros((3, 2, 2))
iris_data = iris.data[:, [1, 3]]  # we take the second and fourth
classes_list = np.unique(iris.target)


def sigma_a():
    class_data = [0] * 3
    iris_mean_0 = np.zeros((len(iris_data), 3))
    for class_number in classes_list:
        class_data[class_number] = iris_data[np.where(iris.target == class_number), :][0]
        means[class_number] = [class_data[class_number][:, 0].mean(), class_data[class_number][:, 1].mean()]
        for row in np.where(iris.target == class_number)[0]:
            iris_mean_0[row] = [iris_data[row][0] - means[class_number][0], iris_data[row][1] - means[class_number][1], class_number]
    sigma_found = np.mean([np.var(iris_mean_0[:, 0]), np.var(iris_mean_0[:, 1])])
    return sigma_found


def plot_a1(sigma_val):
    cov_estim = np.identity(2) * sigma_val
    # PLOTTING RESULTS
    dist_size = 600
    gauss = np.random.multivariate_normal((0, 0), cov_estim, dist_size)
    gauss_classes = [0] * dist_size
    for i, element in enumerate(gauss):
        element_class = i % 3
        gauss_classes[i] = element_class
        gauss[i] = [element[0] + means[element_class][0], element[1] + means[element_class][1]]
    plot_classes(gauss, gauss_classes)


def covariances_b():
    class_data = [0] * 3
    for class_number in classes_list:
        class_data[class_number] = iris_data[np.where(iris.target == class_number), :][0]
        covariances[class_number] = np.cov(class_data[class_number][:, 0], class_data[class_number][:, 1], rowvar=False)
        means[class_number] = [class_data[class_number][:, 0].mean(), class_data[class_number][:, 1].mean()]


def classifier_b(point):
    class_prob = np.zeros(3)
    for class_val in classes_list:
        class_prob[class_val] = stats.multivariate_normal.pdf(point, means[class_val], covariances[class_val])
    return class_prob.argmax()


def parameters_d(prior_probs=None):
    covariances_b()
    common_cov_s = np.zeros((2, 2))
    set_size = len(iris_data)
    class_size = np.zeros(3)
    for class_number in classes_list:
        class_size[class_number] = len(np.where(iris.target == class_number)[0])
        common_cov_s = common_cov_s + covariances[class_number] * class_size[class_number] / set_size

    prior_probs = prior_probs if prior_probs is not None else class_size / set_size
    s_inverted = np.linalg.inv(common_cov_s)
    wi = [np.matmul(s_inverted, means[class_number]) for class_number in classes_list]
    wi0 = [-1 / 2 * np.matmul(np.matmul(np.transpose(means[class_number]), s_inverted), means[class_number]) + np.log(prior_probs[class_number]) for class_number in classes_list]
    return wi, wi0


def classifier_d(point, wi, wi0):
    class_prob = np.zeros(3)
    for class_val in classes_list:
        class_prob[class_val] = np.matmul(wi[class_val], point) + wi0[class_val]
    return class_prob.argmax()


def lines_params_d(prior_probs=None):
    (wi, wi0) = parameters_d(prior_probs)

    dw0_10 = wi0[1] - wi0[0]
    dw0_20 = wi0[2] - wi0[0]
    dw0_21 = wi0[2] - wi0[1]
    dw_10 = [wi[1][0] - wi[0][0], wi[1][1] - wi[0][1]]
    dw_20 = [wi[2][0] - wi[0][0], wi[2][1] - wi[0][1]]
    dw_21 = [wi[2][0] - wi[1][0], wi[2][1] - wi[1][1]]

    lines_params = np.array([
        [-dw_10[0] / dw_10[1], -dw0_10 / dw_10[1]],
        [-dw_20[0] / dw_20[1], -dw0_20 / dw_20[1]],
        [-dw_21[0] / dw_21[1], -dw0_21 / dw_21[1]]
    ])

    return [lines_params[np.argwhere(lines_params[:, 1] == lines_params[:, 1].max())[0]][0], lines_params[np.argwhere(lines_params[:, 1] == lines_params[:, 1].min())[0]][0]]


def point_a():
    sigma = sigma_a()
    print(f"Sigma : {sigma}")
    plot_a1(sigma)


def point_b():
    covariances_b()
    a = [3.2, .2]
    b = [2.7, 1.5]
    c = [3, 2.05]

    print(f"{a} : {classifier_b(a)}")
    print(f"{b} : {classifier_b(b)}")
    print(f"{c} : {classifier_b(c)}")


def point_c():
    covariances_b()
    dist_size = 1500
    random_points = np.random.uniform(size=(dist_size, 2))
    amplify = 1
    h_span = iris_data[:, 0].max() - iris_data[:, 0].min() + amplify
    v_span = iris_data[:, 1].max() - iris_data[:, 1].min() + amplify
    points_lb_corner = [iris_data[:, 0].min() - amplify / 2, iris_data[:, 1].min() - amplify / 2]
    random_points = [point * [h_span, v_span] + points_lb_corner for point in random_points]
    point_classes = [classifier_b(point) for point in random_points]
    plot_classes(np.asarray(random_points), point_classes)


def point_d():
    (wi, wi0) = parameters_d()
    print(wi)
    print(wi0)

    a = [3.2, .2]
    b = [2.7, 1.5]
    c = [3, 2.05]

    print("classifier")
    print(f"{a} : {classifier_d(a, wi, wi0)}")
    print(f"{b} : {classifier_d(b, wi, wi0)}")
    print(f"{c} : {classifier_d(c, wi, wi0)}")

    lines_params = lines_params_d()
    print("lines_params")
    print(lines_params)

    lines = np.zeros((len(lines_params), 2, 2))
    for (idx, line_params) in enumerate(lines_params):
        lines[idx][0] = [0, line_params[1]]
        lines[idx][1] = [5, line_params[0] * 5 + line_params[1]]
    plot_classes(iris_data, iris.target, lines=lines)


def point_e():
    prior_probs = [.2, .3, .5]
    lines_params = lines_params_d(prior_probs)
    print("lines_params")
    print(lines_params)

    lines = np.zeros((len(lines_params), 2, 2))
    for (idx, line_params) in enumerate(lines_params):
        lines[idx][0] = [0, line_params[1]]
        lines[idx][1] = [5, line_params[0] * 5 + line_params[1]]
    plot_classes(iris_data, iris.target, lines=lines)


assignment_header()
point_a()
point_b()
point_c()
point_d()
point_e()

print('done')
