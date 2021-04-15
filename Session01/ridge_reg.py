import numpy as np 
from sklearn.linear_model import Ridge

class ridgeRegression:

    def __init__(self):
        return

    def fit(self, X_train, Y_train, LAMBDA):
        
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]
        W = np.linalg.inv(np.transpose(X_train).dot(X_train) + \
                          LAMBDA * np.identity(X_train.shape[1])
                          ).dot(np.transpose(X_train)).dot(Y_train)
        return W

    def fit_GD(self, X_train, Y_train, 
        LAMBDA, 
        lr, 
        max_num_epochs=100, 
        batch_size=128):
        
        W = np.random.randn(X_train.shape[1])
        lass_lost = 10e+8
        for epoch in range(max_num_epochs):
            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train = X_train[arr]
            Y_train = Y_train[arr]

            total_minibatch = int(np.ceil(X_train.shape[0] / batch_size))
            for i in range(total_minibatch):
                index = i * batch_size
                X_train_sub = X_train[index:index + batch_size]
                Y_train_sub = Y_train[index:index + batch_size]
                
                grad = np.transpose(X_train_sub).dot(X_train_sub.dot(W) - Y_train_sub) + LAMBDA * W
                W = W - lr * grad

            new_loss = self.compute_RSS(self.predict(W, X_train), Y_train)
            if (np.abs(new_loss - lass_lost) <= 1e-5):
                break
            lass_lost = new_loss
        return W

    def predict(self, W, X_new):
        return np.array(X_new).dot(W)

    def compute_RSS(self, Y_new, Y_predicted):
        return 1.0 / Y_new.shape[0] * np.sum((Y_new - Y_predicted) ** 2)

    def get_the_best_LAMBDA(self, X_train, Y_train):

        def cross_validation(num_folds, LAMBDA):

            row_ids = np.array(range(X_train.shape[0]))
            valid_ids = np.split(
                        row_ids[:(len(row_ids) - len(row_ids) % num_folds)], num_folds)
            valid_ids[-1] = np.append(
                        valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [
                        [k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            aver_RSS = 0

            for i in range(num_folds):
                valid_part = {'X': X_train[valid_ids[i]], 'Y': Y_train[valid_ids[i]]}
                train_part = {'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
                W = self.fit(train_part['X'], train_part['Y'], LAMBDA)
                Y_predict = self.predict(W, valid_part['X'])
                aver_RSS += self.compute_RSS(valid_part['Y'], Y_predict)
            return aver_RSS / num_folds

        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    best_LAMBDA = current_LAMBDA
                    minimum_RSS = aver_RSS
            return best_LAMBDA, minimum_RSS

        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=0,
                                              minimum_RSS=10000 ** 2,
                                              LAMBDA_values = range(50))
        LAMBDA_values = [k * 1. / 1000 for k in range(
                        max(0, (best_LAMBDA -1)*1000), (best_LAMBDA + 1)*1000, 1)]

        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=best_LAMBDA,
                                              minimum_RSS=minimum_RSS,
                                              LAMBDA_values = LAMBDA_values)

        return best_LAMBDA

    def get_the_best_lr(self, X_train, Y_train, LAMBDA):

        def cross_validation(num_folds, lr):

            row_ids = np.array(range(X_train.shape[0]))
            valid_ids = np.split(
                row_ids[:(len(row_ids) - len(row_ids) % num_folds)], num_folds)
            valid_ids[-1] = np.append(
                valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [
                [k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            aver_RSS = 0

            for i in range(num_folds):
                valid_part = {'X': X_train[valid_ids[i]], 'Y': Y_train[valid_ids[i]]}
                train_part = {'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
                W = self.fit_GD(train_part['X'], train_part['Y'], LAMBDA, lr)
                Y_predict = self.predict(W, valid_part['X'])
                aver_RSS += self.compute_RSS(valid_part['Y'], Y_predict)
            return aver_RSS / num_folds

        def range_scan(best_lr, minimum_RSS, lr_values):
            for current_lr in lr_values:
                aver_RSS = cross_validation(num_folds=5, lr=current_lr)
                if aver_RSS < minimum_RSS:
                    best_lr = current_lr
                    minimum_RSS = aver_RSS
            return best_lr, minimum_RSS

        best_lr, minimum_RSS = range_scan(best_lr=0,
                                              minimum_RSS=10000 ** 2,
                                              lr_values = range(50))
        lr_values = [k * 1. / 1000 for k in range(
                        max(0, (best_lr -1)*1000), (best_lr + 1)*1000, 1)]

        best_lr, minimum_RSS = range_scan(best_lr=best_lr,
                                              minimum_RSS=minimum_RSS,
                                              lr_values = lr_values)

        return best_lr
        
#Load Data
def get_data(path):
    df = np.loadtxt(path)
    X = np.array(df[:,1:-1])
    Y = np.array(df[:,-1])
    return X, Y

#Normalize Data
def normalize_and_add_ones(X):
    X_max = np.array([[np.amax(X[:, column_id])
                       for column_id in range(X.shape[1])]
                      for _ in range(X.shape[0])])
    X_min = np.array([[np.amin(X[:, column_id])
                       for column_id in range(X.shape[1])]
                      for _ in range(X.shape[0])])

    X_normalized = (X - X_min) / (X_max - X_min)

    ones = np.array([[1] for _ in range(X.shape[0])])
    return np.column_stack((ones, X_normalized))       

if __name__ == '__main__':
    X, Y = get_data(path = "./death_rate_data.txt")  
    # Normalization
    X = normalize_and_add_ones(X)
    #Train_test_split
    X_train, Y_train = X[:50], Y[:50]
    X_test, Y_test = X[50:], Y[50:]

    model = ridgeRegression()
    best_LAMBDA = model.get_the_best_LAMBDA(X_train, Y_train)
    print("Ridge Regression closed-form solution:")
    print('Best LAMBDA: ', best_LAMBDA)
    W_learned = model.fit(X_train = X_train, Y_train = Y_train, LAMBDA = best_LAMBDA)
    Y_predicted = model.predict(W=W_learned, X_new=X_test)
    print('Loss: ',model.compute_RSS(Y_new=Y_test, Y_predicted=Y_predicted))
    print("----------------------------------------------------")

    print("Implement with scikit-learn:")
    ridge_reg = Ridge(alpha = best_LAMBDA, solver='cholesky')
    ridge_reg.fit(X_train, Y_train)
    print('Loss: ', model.compute_RSS(Y_test, ridge_reg.predict(X_test)))
    print("-----------------------------------------------------")

    print("Using Gradient Descent")
    best_lr = model.get_the_best_lr(X_train, Y_train, best_LAMBDA)
    print('Best Learning Rate: ', best_lr)
    W_learned = model.fit_GD(X_train = X_train, Y_train = Y_train, LAMBDA = best_LAMBDA, lr = best_lr)
    Y_predicted = model.predict(W=W_learned, X_new=X_test)
    print('Loss: ',model.compute_RSS(Y_new=Y_test, Y_predicted=Y_predicted))
    print("-----------------------------------------------------")
    
    print("Implement with scikit-learn:")
    ridge_reg = Ridge(alpha = best_LAMBDA, solver='sparse_cg')
    ridge_reg.fit(X_train, Y_train)
    print('Loss: ', model.compute_RSS(Y_test, ridge_reg.predict(X_test)))

