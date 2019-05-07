import numpy as np
import matplotlib.pyplot as plt


def bivariate_gauss(means, cov_matrix):
    x, y = np.random.multivariate_normal(means, cov_matrix, 5000).T
    x_min, x_max = x.min() - .5, x.max() + .5
    y_min, y_max = y.min() - .5, y.max() + .5

    plt.figure(figsize=(8, 6))
    
    # Plot the training points
    plt.scatter(x, y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


mean = (1, 10)
cov = [[1, 0], [0, 1]]
bivariate_gauss(mean, cov)
