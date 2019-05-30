import numpy as np
from scipy import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, chi2_kernel
from sklearn.svm import SVC

# def point_a()
spanish_words_file = "./spanish_words_500.data"
french_words_file = "./french_words_500.data"
misclassified_words = {}

with open(spanish_words_file, 'r') as words_file:
    spanish_words = [line.replace('\n', '') for line in words_file.readlines()]
with open(french_words_file, 'r') as words_file:
    french_words = [line.replace('\n', '') for line in words_file.readlines()]

# spanish_words = spanish_words[0:100]
# french_words = french_words[0:100]

words_data = np.append(spanish_words, french_words)
words_classes = np.append(np.full(len(spanish_words), 'S'), np.full(len(french_words), 'F'))

training_fraction = 0.7
words_idxs = [i for i in range(len(words_data))]
random.shuffle(words_idxs)
training_size = int(len(words_idxs) * training_fraction)
words_training = words_data[words_idxs[0:training_size]]
words_training_classes = words_classes[words_idxs[0:training_size]]
words_test = words_data[words_idxs[training_size - 1:len(words_data)]]
words_test_classes = words_classes[words_idxs[training_size - 1:len(words_data)]]

print(f"Entire  dataset size: {len(words_data)}")
print(f"Training set size: {training_size}")
print(f"Test set size: {len(words_data) - training_size}")

# def point_b()
words_vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 1))
# words_data = [re.sub(r'(.)', r'\1 ', word) for word in words_data]
words_term_document = words_vectorizer.fit_transform(words_training)
feature_names = words_vectorizer.get_feature_names()
print("n-grams features:")
print(f"Size: {len(feature_names)}")
print(f"Features:\n{feature_names}")


def cosine_kernel(word_one, word_two, vectorizer=words_vectorizer):
    word_one_terms, word_two_terms = tuple(vectorizer.transform([word_one, word_two]).toarray())
    return cosine_similarity([word_one_terms], [word_two_terms])[0][0]


def intersection_kernel(word_one, word_two, vectorizer=words_vectorizer):
    word_one_terms, word_two_terms = tuple(vectorizer.transform([word_one, word_two]).toarray())
    word_one_terms, word_two_terms = (word_one_terms / word_one_terms.sum(), word_two_terms / word_two_terms.sum())
    intersection = [letter_term if letter_term < word_two_terms[idx] else word_two_terms[idx] for idx, letter_term in enumerate(word_one_terms)]
    return np.sum(intersection)


def chi_square_kernel(word_one, word_two, vectorizer=words_vectorizer):
    word_one_terms, word_two_terms = tuple(vectorizer.transform([word_one, word_two]).toarray())
    return chi2_kernel([word_one_terms], [word_two_terms])[0][0]


def ssk_kernel(word_one, word_two, vectorizer=words_vectorizer):
    max_substring = vectorizer.ngram_range[1]
    return ssk(word_one, word_two, max_substring, .5, accum=True)


# point b.i
print("\nComparing kernel result for words [gato, perro] (1), [gato, gatto] (2), and [gato, gata](3)\n")
print("\ncosine_kernel")
print(f'1. {cosine_kernel("gato", "perro")}')
print(f'2. {cosine_kernel("gato", "gatto")}')
print(f'3. {cosine_kernel("gato", "gata")}')

# point b.ii
print("\nintersection_kernel")
print(f'1. {intersection_kernel("gato", "perro")}')
print(f'2. {intersection_kernel("gato", "gatto")}')
print(f'3. {intersection_kernel("gato", "gata")}')

# point b.iii
print("\nchi_square_kernel")
print(f'1. {chi_square_kernel("gato", "perro")}')
print(f'2. {chi_square_kernel("gato", "gatto")}')
print(f'3. {chi_square_kernel("gato", "gata")}')

# point b.iv
print("\nssk_kernel")
print(f'1. {chi_square_kernel("gato", "perro")}')
print(f'2. {chi_square_kernel("gato", "gatto")}')
print(f'3. {chi_square_kernel("gato", "gata")}')


# def point_c()
def make_training_kernel_matrix(kernel, training_data, vectorizer=words_vectorizer):
    kernel_matrix = np.zeros((len(training_data), len(training_data)))
    for idx, data_col in enumerate(training_data):
        kernel_matrix[idx][idx] = kernel(data_col, data_col, vectorizer=vectorizer)
        for jdx, data_row in enumerate(training_data):
            if jdx > idx:
                kernel_val = kernel(data_col, data_row, vectorizer=vectorizer)
                kernel_matrix[idx][jdx] = kernel_val
                kernel_matrix[jdx][idx] = kernel_val
    return kernel_matrix


def make_kernel_matrix(kernel, test_data, training_data, vectorizer=words_vectorizer):
    kernel_matrix = np.zeros((len(test_data), len(training_data)))
    for idx, data_col in enumerate(test_data):
        for jdx, data_row in enumerate(training_data):
            kernel_matrix[idx][jdx] = kernel(data_col, data_row, vectorizer=vectorizer)
    return kernel_matrix


def cross_validation_scores(ngram_maxs, kernels, complexities):
    scores = np.zeros((len(ngram_maxs), len(kernels), len(complexities), 2))
    for idx, ngram_max in enumerate(ngram_maxs):
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, ngram_max))
        vectorizer.fit(words_training)
        features = vectorizer.get_feature_names()
        print(f"n-gram: (1,{ngram_max})")
        print(f"Number of features: {len(features)}")
        print(f"Features:\n{features}")
        scores[idx] = [cross_validation_complexities(vectorizer, kernel, complexities) for kernel in kernels]
    return scores


def cross_validation_complexities(vectorizer, kernel, complexities):
    kernel_training = make_training_kernel_matrix(kernel, words_training, vectorizer)
    kernel_test = make_kernel_matrix(kernel, words_test, words_training, vectorizer)
    return cross_validation_complexities_precomputed(kernel_training, kernel_test, complexities)


def cross_validation_complexities_precomputed(kernel_training, kernel_test, complexities):
    scores = np.zeros((len(complexities), 2))
    for idx, complexity in enumerate(complexities):
        svm = SVC(kernel='precomputed', C=complexity)
        svm.fit(kernel_training, words_training_classes)
        prediction_training = svm.predict(kernel_training)
        prediction_test = svm.predict(kernel_test)
        # score_training = svm.score(kernel_training, words_training_classes)
        # score_test = svm.score(kernel_test, words_test_classes)

        score_test, score_training = compute_scores(prediction_test, prediction_training)
        scores[idx] = [score_training, score_test]
    print(f"scores: {scores}")
    return scores


def compute_scores(prediction_test, prediction_training):
    differences_training = prediction_training == words_training_classes
    differences_test = prediction_test == words_test_classes
    score_training = differences_training.sum() / len(differences_training)
    score_test = differences_test.sum() / len(differences_test)
    misclassified = np.append(words_training[differences_training], words_test[differences_test])
    for word in misclassified:
        misclassified_words[word] = 1 if word not in misclassified_words else misclassified_words[word] + 1
    return score_test, score_training


ngram_maxs = [1, 2, 3]
kernels = [cosine_kernel, intersection_kernel, chi_square_kernel]
kernel_names = ['cosine_kernel', 'intersection_kernel', 'chi_square_kernel']
complexities = [2 ** -15, 2 ** -10, 2 ** -5, 2, 2 ** 5]
cross_validation = cross_validation_scores(ngram_maxs, kernels, complexities)

# FOR SSK
decay = 0.5
max_substrings = [1, 2, 3]
cross_validation_ssk = np.zeros((len(max_substrings),len(complexities),2))
words_training_t = np.transpose([words_training])
words_test_t = np.transpose([words_test])
for idx, max_substring in enumerate(max_substrings):
    kernel_training = string_kernel(words_training_t, words_training_t, max_substring, decay)
    kernel_test = string_kernel(words_test_t,words_training_t, max_substring, decay)
    cross_validation_ssk[idx] = cross_validation_complexities_precomputed(kernel_training, kernel_test, complexities)


# def point_d()
class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in
        IPython Notebook. """

    def _repr_html_(self):
        html = ["<table>"]
        for ridx, row in enumerate(self):
            html.append("<tr>")
            for col in row:
                html.append(f"<td>{col}</td>" if ridx != 0 else f"<th>{col}</th>")
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


# point d.i
scores_table = ListTable()
scores_table.append(['Kernel', 'Regulation parameter', 'Max. n-gram', 'Score training set', 'Score test set'])
for n, ngram in enumerate(cross_validation):
    for k, kernel in enumerate(ngram):
        for c, complexity_scores in enumerate(kernel):
            scores_table.append([kernel_names[k], f"2^{int(np.log2(complexities[c]))}", ngram_maxs[n], f"{complexity_scores[0] * 100:.2f}%", f"{complexity_scores[1] * 100:.2f}%"])

for n, max_substring in enumerate(cross_validation_ssk):
    for c, complexity_scores in enumerate(max_substring):
        scores_table.append(['ssk_kernel', f"2^{int(np.log2(complexities[c]))}", max_substrings[n], f"{complexity_scores[0] * 100:.2f}%", f"{complexity_scores[1] * 100:.2f}%"])

print("Table 3.d.i.")
print(scores_table)

# point d.ii
misclassification_table = ListTable()
misclassification_table.append(['#', 'Word', 'Language', 'Misclassifications'])
misclassified_top = sorted(misclassified_words, key=misclassified_words.get, reverse=True)[0:40]
for word in misclassified_top:
    misclassification_table.append([len(misclassification_table), word, 'ES' if word in spanish_words else 'FR', misclassified_words[word]])
print("Table 3.d.ii.a.")
print(misclassification_table)

french_mc = np.sum([misclassified_words[key] for key in misclassified_words if key in french_words])
spanish_mc = np.sum([misclassified_words[key] for key in misclassified_words if key in spanish_words])
miscl_by_lang_table = ListTable()
miscl_by_lang_table.append(['Language', 'Total misclassifications', 'Porcentual misclassifications'])
miscl_by_lang_table.append(['Spanish', spanish_mc, f"{spanish_mc / (spanish_mc + french_mc) * 100:.2f}%"])
miscl_by_lang_table.append(['French', french_mc, f"{french_mc / (spanish_mc + french_mc) * 100:.2f}%"])
print("Table 3.d.ii.b.")
print(miscl_by_lang_table)

print('done')
