from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.interactive(False)
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

iris_hist = iris_df['sepal length (cm)'].hist(bins=30)
plt.show()


iris_df.drop([iris.feature_names[0], iris.feature_names[2]], 1)
for class_number in np.unique(iris.target):
    plt.figure(1)
    iris_df['sepal length (cm)'].iloc[np.where(iris.target == class_number)[0]].hist(bins=30)
plt.show()

iris_df_with_class = pd.DataFrame(np.column_stack((iris.data, iris.target)), columns=np.append(iris.feature_names, ['class']))
iris_df_filtered = iris_df_with_class[iris_df_with_class['class'].isin([0, 1])]
print('end')
