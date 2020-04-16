import random as rd

import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC


# @gamma: The Gamma of svm.SVC
# @layers: The number of layers (the repeats in the method)
# @number_of_partitions: A vector of partitions to be done in each layer
def CascadeSVM(data, target, kernel, gamma, number_of_partitions,
               number_of_layers,
               counting=1, C=None, use_LinearSVC=False):
    if number_of_layers != 0:
        # We generate a random order of the sample
        random_order = rd.sample(range(len(data)), len(data))

        # Pick a number multiple of len(train_images_svm)=60000
        size_subsample = np.int(len(data) / number_of_partitions[0])

        support_vectors = []
        for partition in range(number_of_partitions[0]):
            first = partition * size_subsample
            last = (partition + 1) * size_subsample
            values = random_order[first:last]
            data_partition = data[values, :]
            target_partition = target[values]

            if C is None:
                # Let's evlauate different C's, UNCOMMENT THIS CODE IF YOU WANT TO PICK A C USING CV
                c = np.logspace(-5, 4, 10, base=2)
                c_score = []
                for c_value in c:
                    if kernel != 'linear' or (not use_LinearSVC):
                        clf = svm.SVC(C=c, gamma=gamma, kernel=kernel)
                    else:
                        clf = LinearSVC(C=c)
                    score = cross_val_score(clf, data_partition, target_partition, cv=5)
                    c_score.append(score.mean())
                C = c[np.argmax(c_score)]

            clf = svm.SVC(C=C, gamma=gamma, kernel=kernel)
            clf.fit(data_partition, target_partition)
            vectors = clf.support_ + first
            support_vectors = np.r_[support_vectors, vectors]
            # print('Layer ' + np.str(counting) + ' and Partition ' + np.str(partition+1) )

        # Create the new data,target, layers and partitions
        support_vectors = support_vectors.astype(int)

        random_order = np.array(random_order)
        new_data = data[random_order[support_vectors], :]
        new_target = target[random_order[support_vectors]]
        new_number_of_layers = number_of_layers - 1
        new_number_of_partitions = np.delete(number_of_partitions, 0)
        print('The layer No.' + np.str(counting) + ' (with ' + np.str(number_of_partitions[0]) +
              ' partitions)' +
              ' has been completed with ' +
              np.str(len(support_vectors)) + ' support vectors')

        return CascadeSVM(data=new_data, target=new_target, number_of_layers=new_number_of_layers,
                          kernel=kernel, gamma=gamma, number_of_partitions=new_number_of_partitions,
                          counting=counting + 1, C=C, use_LinearSVC=use_LinearSVC)
    else:
        if C == None:
            # Let's evlauate different C's, UNCOMMENT THIS CODE IF YOU WANT TO PICK A C USING CV
            c = np.logspace(-5, 4, 10, base=2)
            c_score = []
            for c_value in c:
                if kernel != 'linear' or (not use_LinearSVC):
                    clf = svm.SVC(C=c, gamma=gamma, kernel=kernel)
                else:
                    clf = LinearSVC(C=c)
                score = cross_val_score(clf, data, target, cv=5)
                c_score.append(score.mean())
            C = c[np.argmax(c_score)]
        if kernel != 'linear' or (not use_LinearSVC):
            clf = svm.SVC(C=C, gamma=gamma, kernel=kernel)
            print('You pick to use regular svm.SVC')
        else:
            clf = LinearSVC(C=C)
            print('You pick to use Linear_SVC')

        clf.fit(data, target)
        return clf
