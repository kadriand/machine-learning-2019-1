import numpy as np
from sklearn import datasets
from assignment2.commons import plot_classes, iris_confusion_matrix
from scipy import stats, random

iris = datasets.load_iris()
means = np.zeros((3, 2))
covariances = np.zeros((3, 2, 2))
iris_data = iris.data[:, [1, 3]]  # we take the second and fourth
classes_list = np.unique(iris.target)
iris_idxs = [i for i in range(len(iris_data))]


def make_random_covariances_3():
    for class_number in classes_list:
        random_matrix = random.rand(2, 2)
        random_positive_semidefinite = np.dot(random_matrix, random_matrix.transpose())
        covariances[class_number] = random_positive_semidefinite
        print(f"\nRandom covariance matrix for class {class_number} '{iris.target_names[class_number]}': \n {covariances[class_number]}")


def find_means_3(data_set=iris_data, data_classes=iris.target):
    class_data = [0] * 3
    for class_number in classes_list:
        class_data[class_number] = data_set[np.where(data_classes == class_number), :][0]
        means[class_number] = [class_data[class_number][:, 0].mean(), class_data[class_number][:, 1].mean()]
        print(f"\nMean for  class {class_number} '{iris.target_names[class_number]}':\n {means[class_number]}")


def classifier_3_c(point, loss):
    class_prob = np.zeros(4)
    for class_val in classes_list:
        class_prob[class_val] = stats.multivariate_normal.pdf(point, means[class_val], covariances[class_val])
    class_prob[3] = 1 - loss
    return class_prob.argmax()


def point_a():
    find_means_3()
    samples_per_class = 200
    dist_points = np.zeros((samples_per_class * len(classes_list), 2))
    dist_classes = np.zeros(samples_per_class * len(classes_list))
    for class_number in classes_list:
        gauss = np.random.multivariate_normal(means[class_number], covariances[class_number], samples_per_class)
        for i, point in enumerate(gauss):
            idx = i + class_number * samples_per_class
            dist_classes[idx] = class_number
            dist_points[idx] = point
    plot_classes(dist_points, dist_classes, "Distribution random covariances matrices")


def point_b():
    random.shuffle(iris_idxs)
    samples_size = int(len(iris_data) * 0.8)
    iris_training = iris_data[iris_idxs[0:samples_size]]
    iris_training_classes = iris.target[iris_idxs[0:samples_size]]
    print("DISTRIBUTION PARAMETERS")
    for class_number in classes_list:
        print(f"\nCovariance matrix for class {class_number} '{iris.target_names[class_number]}': \n {covariances[class_number]}")
    find_means_3(iris_training, iris_training_classes)
    plot_classes(iris_training, iris_training_classes, "Training set 80%")


def point_c():
    samples_size = int(len(iris_data) * 0.8)
    loss = 0.8
    iris_test = iris_data[iris_idxs[samples_size - 1:len(iris_data)]]
    iris_test_classes = [classifier_3_c(point, loss) for point in iris_test]
    plot_classes(iris_test, iris_test_classes, f"Remaining set 20% with loss of \u03bb={loss}")

    iris_test_real_classes = iris.target[iris_idxs[samples_size - 1:len(iris_data)]]
    iris_confusion_matrix(iris_test_real_classes, np.asarray(iris_test_classes), f"Confusion matrix with loss of \u03bb={loss}")


# assignment_header()
make_random_covariances_3()
point_a()
point_b()
point_c()

print('done')
