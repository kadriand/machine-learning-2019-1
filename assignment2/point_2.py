import numpy as np
from sklearn import datasets
from assignment2.commons import plot_classes, assignment_header
import scipy.stats as stats

iris = datasets.load_iris()
y = iris.target
means = np.zeros((3, 2))
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


def plot_2_a(covariance_matrix):
    # PLOTTING RESULTS
    dist_size = 600
    gauss = np.random.multivariate_normal((0, 0), covariance_matrix, dist_size)
    gauss_classes = [0] * dist_size
    for i, element in enumerate(gauss):
        element_class = i % 3
        gauss_classes[i] = element_class
        gauss[i] = [element[0] + means[element_class][0], element[1] + means[element_class][1]]
    plot_classes(gauss, gauss_classes)


def classifier_2(point, loss, covariance_matrix):
    class_prob = np.zeros(4)
    for class_val in classes_list:
        class_prob[class_val] = stats.multivariate_normal.pdf(point, means[class_val], covariance_matrix)
    class_prob[3] = 1 - loss
    return class_prob.argmax()


def point_a():
    sigma = sigma_a()
    diagonal_covariance = np.identity(2) * sigma
    print(f"Sigma :\n {sigma}")
    print(f"Covariances Matrix :\n {diagonal_covariance}")
    plot_2_a(diagonal_covariance)


def point_b():
    diagonal_covariance = np.identity(2) * sigma_a()
    a = [3.2, .2]
    b = [2.7, 1.5]
    c = [3, 2.05]
    loss_lambda = 0.5

    print(f"{a} : {classifier_2(a, loss_lambda, diagonal_covariance)}")
    print(f"{b} : {classifier_2(b, loss_lambda, diagonal_covariance)}")
    print(f"{c} : {classifier_2(c, loss_lambda, diagonal_covariance)}")


def point_c():
    diagonal_covariance = np.identity(2) * sigma_a()
    loss_lambdas = [.3, .5, .9, 1]
    dist_size = 1000
    random_points = np.random.uniform(size=(dist_size, 2))
    amplify = 1
    h_span = iris_data[:, 0].max() - iris_data[:, 0].min() + amplify
    v_span = iris_data[:, 1].max() - iris_data[:, 1].min() + amplify
    points_lb_corner = [iris_data[:, 0].min() - amplify / 2, iris_data[:, 1].min() - amplify / 2]
    random_points = [point * [h_span, v_span] + points_lb_corner for point in random_points]

    for loss_lambda in loss_lambdas:
        point_classes = [classifier_2(point, loss_lambda, diagonal_covariance) for point in random_points]
        plot_classes(np.asarray(random_points), point_classes, title=f"Loss \u03bb= {loss_lambda}")


assignment_header()
point_a()
point_b()
point_c()

print('done')
