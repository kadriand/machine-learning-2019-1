from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

iris = datasets.load_iris()


def plot_classes(points, classes, title=None, lines=[]):
    x_min, x_max = points[:, 0].min() - .5, points[:, 0].max() + .5
    y_min, y_max = points[:, 1].min() - .5, points[:, 1].max() + .5
    plt.figure(3, figsize=(8, 6))
    plt.clf()
    # Plot the training points
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c=classes, cmap=plt.cm.Set1, edgecolor='k')
    axis_labels = [i for j, i in enumerate(iris.feature_names) if j in [1, 3]]
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    ax.grid(True)
    if len(lines) > 0:
        for serie in lines:
            plt.plot(serie[:, 0], serie[:, 1], '-o')
    if title:
        plt.title(title)
    plt.show()


def assignment_header():
    iris_data = iris.data[:, [1, 3]]  # we take the second and fourth
    iris_classes = iris.target
    plot_classes(iris_data, iris_classes, title="Original dataset")


def iris_confusion_matrix(real, predicted, title="Confusion matrix"):
    classes_list = np.append(np.unique(iris.target), 3)
    confusion_matrix = metrics.confusion_matrix(real, predicted, classes_list)
    print(f"confusion_matrix:\n {confusion_matrix}")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix)
    plt.title(title)
    fig.colorbar(cax)
    axis = np.append([''], np.append(iris.target_names, ['rejected']))
    ax.set_xticklabels(axis)
    ax.set_yticklabels(axis)
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.show()
