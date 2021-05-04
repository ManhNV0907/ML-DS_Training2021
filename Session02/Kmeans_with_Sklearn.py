import numpy as np 
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from collections import defaultdict

#Implement with Scikit-learn
#Kmeans
def load_data(data_path, vocab_path):
    data = []
    labels = []
    def sparse_to_dense(sparse_r_d, vocab_size):
        #Convert sparse to vector
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)
    
    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open(vocab_path) as f:
        vocab_size = len(f.read().splitlines())

    
    for data_id, d in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        labels.append(label)
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        data.append(r_d)
    return data, labels
def cluster(data):
    # Use csr_matrix to create a sparse matrix with efficient row slicing
    X = csr_matrix(data)
    print("________________________________________")
    kmeans = KMeans(
        n_clusters = 20,
        init = 'random',
        n_init = 5, # Number of time that Kmeans runs with diffrently initialized centroids
        tol = 1e-3, # Threshold for acceptable minimum error decrease
        random_state = 2018 # Set to get determinstic results
    ).fit(X)

    labels_pred = kmeans.labels_
    return labels_pred

def compute_purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def compute_NMI(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)

if __name__ == "__main__":
    data_path = "./datasets/20news-bydate/data_tf_idf.txt"
    vocab_path = "./datasets/20news-bydate/words_idfs.txt"
    data, labels = load_data(data_path, vocab_path)    
    clustering_labels = cluster(data)
    print("Purity Score: ", compute_purity(labels, clustering_labels))
    print("NMI score: ", compute_NMI(labels, clustering_labels))
    


    
