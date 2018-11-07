

import pickle
import numpy as np
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from dbn.models import UnsupervisedDBN 


n_epochs_rbm = 1   # 20
logistic_inverse_reg = 50.0  # 6000.0
logistic_inverse_reg_2 = 1   # 100.0


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

# Load Data
digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression(solver='newton-cg',
                                           multi_class='auto',
                                           C=logistic_inverse_reg)
dbn = UnsupervisedDBN(hidden_layers_structure=[256, 512],
                      batch_size=10,
                      learning_rate_rbm=0.06,
                      n_epochs_rbm=n_epochs_rbm,
                      activation_function='sigmoid')

classifier = Pipeline(steps=[('dbn', dbn),
                             ('logistic', logistic)])

# Training RBM-logistic pipeline
classifier.fit(X_train, Y_train)

# Training logistic regression
logistic_classifier = linear_model.LogisticRegression(solver='newton-cg',
                                                      multi_class='auto',
                                                      C=logistic_inverse_reg_2)
logistic_classifier.fit(X_train, Y_train)

# Save model
with open('logistic.pkl', 'wb') as wf:
    pickle.dump(classifier, wf)

# Evaluation
print("\nLogistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))
print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))

