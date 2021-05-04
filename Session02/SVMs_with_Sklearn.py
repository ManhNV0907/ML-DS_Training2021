import numpy as np 
from sklearn.svm import LinearSVC
from Kmeans_with_Sklearn import load_data

def compute_accuracy(y_predicted, y_expected):
    matches = np.equal(y_predicted, y_expected)
    accuracy = np.sum(matches.astype(float)) / len(y_expected)
    return accuracy

#LinearSVC    
def classifying_with_linear_SVMs():
    X, y =load_data(data_path = "./datasets/20news-bydate/data_tf_idf.txt", vocab_path="./datasets/20news-bydate/words_idfs.txt")
    X_train, y_train = X[:14000], y[:14000]
    X_test, y_test = X[14000:], y[14000:]
    classifier = LinearSVC(
        C = 10.0,
        tol = 0.001, # Tolerance for stopping criteria
        verbose = True # Whether prints out logs or not
    )
    
    classifier.fit(X_train, y_train)
    y_predicted = classifier.predict(X_test)
    accuracy = compute_accuracy(y_predicted, y_expected = y_test)
    print("Accuracy: {}".format(accuracy))

if __name__ == '__main__':
    classifying_with_linear_SVMs()