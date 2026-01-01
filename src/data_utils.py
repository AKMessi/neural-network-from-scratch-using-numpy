import pickle
import numpy as np
import os

def load_cifar10(root_dir='./data/cifar10'):
    """
    Loads CIFAR-10 dataset.
    """

    def load_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.astype("float")
            Y = np.array(Y)
            return X, Y
        
    xs = []
    ys = []

    print(f"Loading data from {root_dir}...")

    for b in range(1, 6):
        f = os.path.join(root_dir, 'data_batch_%d' % (b, ))
        if not os.path.exists(f):
            raise FileNotFoundError(f"Could not find {f}. Check your paths!")
        
        X, Y = load_batch(f)
        xs.append(X)
        ys.append(Y)

    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    test_path = os.path.join(root_dir, 'test_batch')

    X_test, y_test = load_batch(test_path)

    mean_image = np.mean(X_train, axis=0)

    X_train -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_test, y_test
