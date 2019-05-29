import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy import random
from sklearn.svm import LinearSVC

digits = datasets.load_digits()
digits_data = digits.data
print(f"Data shape: {digits.data.shape}")
training_fraction = 0.7


def normalize_b():
    digits_mean_zero = digits_data - digits_data.mean()
    sum_digits_sq = np.multiply(digits_mean_zero, digits_mean_zero).sum()
    factor = np.sqrt(len(digits_data) * len(digits_data[0]) / sum_digits_sq)
    digits_std_one = digits_mean_zero * factor
    return digits_std_one


def filter_classes_c(class_one, class_two):
    digits_std_one = normalize_b()
    classes_idx = np.append(np.where(digits.target == class_one), np.where(digits.target == class_two))
    digits_data_filtered = digits_std_one[classes_idx, :]
    digits_classes_filtered = digits.target[classes_idx]
    return digits_data_filtered, digits_classes_filtered


def classify_c(training_data, training_classes, test_data, test_classes, complexities):
    training_errors = np.zeros(len(complexities))
    test_errors = np.zeros(len(complexities))
    for idx, complexity in enumerate(complexities):
        linear_svm = LinearSVC(C=complexity, max_iter=5000, tol=1e-5)
        linear_svm.fit(training_data, training_classes)
        training_errors[idx] = 1.0 - linear_svm.score(training_data, training_classes)
        test_errors[idx] = 1.0 - linear_svm.score(test_data, test_classes)

    return training_errors, test_errors


def optimal_C_c(complexities_C, digit_1, digit_2):
    digits_filtered, digits_filtered_classes = filter_classes_c(digit_1, digit_2)
    digits_idxs = [i for i in range(len(digits_filtered))]
    random.shuffle(digits_idxs)
    samples_size = int(len(digits_idxs) * training_fraction)

    digits_training = digits_filtered[digits_idxs[0:samples_size]]
    digits_training_classes = digits_filtered_classes[digits_idxs[0:samples_size]]
    digits_test = digits_filtered[digits_idxs[samples_size - 1:len(digits_filtered)]]
    digits_test_classes = digits_filtered_classes[digits_idxs[samples_size - 1:len(digits_filtered)]]

    training_errors, test_errors = classify_c(digits_training, digits_training_classes, digits_test, digits_test_classes, complexities_C)
    optimal_idx = (test_errors + training_errors).argmin()
    return optimal_idx, test_errors, training_errors, digits_idxs


def plot_regularization_parameters_c(complexities_C, optimal_C, test_errors, training_errors, title=f"Cross Validation Plot"):
    plt.plot(complexities_C, training_errors, label='Training set')
    plt.plot(complexities_C, test_errors, label='Test set')
    max_val = test_errors.max() if test_errors.max() > training_errors.max() else training_errors.max()
    plt.plot([optimal_C, optimal_C], [0, max_val], linestyle=':', label=f'Optimal C: 2^{int(np.log2(optimal_C))}')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Error')
    plt.title(title)
    plt.xscale('log', basex=2)
    plt.legend()
    plt.show()


def weights_vector_d(digits_idxs, optimal_C, digit_1, digit_2):
    digits_filtered, digits_filtered_classes = filter_classes_c(digit_1, digit_2)
    samples_size = int(len(digits_idxs) * training_fraction)
    digits_training = digits_filtered[digits_idxs[0:samples_size]]
    digits_training_classes = digits_filtered_classes[digits_idxs[0:samples_size]]
    linear_svm = LinearSVC(C=optimal_C, max_iter=5000, tol=1e-5)
    linear_svm.fit(digits_training, digits_training_classes)
    weights_vector = linear_svm.coef_[0]
    return weights_vector


def digits_classifier_f(complexities_C, digit_1, digit_2):
    print(f"Digits {digit_1} and {digit_2}")
    optimal_idx, test_errors, training_errors, digits_idxs = optimal_C_c(complexities_C, digit_1, digit_2)
    optimal_C = complexities_C[optimal_idx]
    print(f"Optimal C: {optimal_C}")
    plot_regularization_parameters_c(complexities_C, optimal_C, test_errors, training_errors, title=f"Cross Validation Plot. Digits {digit_1} and {digit_2}")
    weights_vector = weights_vector_d(digits_idxs, optimal_C, digit_1, digit_2)
    print(f"Weights vector:\n {weights_vector}")
    weights_mesh = weights_vector.reshape((8, 8))
    plt.pcolor(weights_mesh, cmap='bwr')
    plt.title(f"Color map. Digits {digit_1} and {digit_2}")
    plt.show()


def point_a():
    plt.gray()
    plt.matshow(digits.images[0])
    plt.show()


def point_b():
    print(f"Original data min: {digits_data.min()}")
    print(f"Original data max: {digits_data.max()}")
    print(f"Original data mean: {digits_data.mean()}")
    print(f"Original data std: {digits_data.std()}")
    digits_std_one = normalize_b()
    print(f"Normalized mean: {digits_std_one.mean()}")
    print(f"Normalized std: {digits_std_one.std()}")


def point_c():
    complexities_C = np.logspace(-15.0, 10.0, num=26, base=2)
    optimal_idx, test_errors, training_errors, digits_idxs = optimal_C_c(complexities_C, 5, 8)
    optimal_C = complexities_C[optimal_idx]
    print(f"Optimal C: {optimal_C}")
    plot_regularization_parameters_c(complexities_C, optimal_C, test_errors, training_errors)


def point_d():
    complexities_C = np.logspace(-15.0, 10.0, num=26, base=2)
    optimal_idx, test_errors, training_errors, digits_idxs = optimal_C_c(complexities_C, 5, 8)
    optimal_C = complexities_C[optimal_idx]
    weights_vector = weights_vector_d(digits_idxs, optimal_C, 5, 8)
    print(f"Weights vector:\n {weights_vector}")


def point_e():
    complexities_C = np.logspace(-15.0, 10.0, num=26, base=2)
    optimal_idx, test_errors, training_errors, digits_idxs = optimal_C_c(complexities_C, 5, 8)
    optimal_C = complexities_C[optimal_idx]
    weights_vector = weights_vector_d(digits_idxs, optimal_C, 5, 8)
    #  POINT E.i
    weights_mesh = weights_vector.reshape((8, 8))
    #  POINT E.ii
    plt.pcolor(weights_mesh)
    plt.show()
    #  POINT E.iii
    plt.pcolor(weights_mesh, cmap='bwr')
    plt.title("Color map")
    plt.show()


def point_f():
    complexities_C = np.logspace(-20.0, 10.0, num=101, base=2)
    digits_classifier_f(complexities_C, 0, 6)
    digits_classifier_f(complexities_C, 0, 9)
    digits_classifier_f(complexities_C, 3, 8)
    digits_classifier_f(complexities_C, 4, 9)


point_a()
point_b()
point_c()
point_d()
point_e()
point_f()

print('done')
