import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural network.
        
        :param self: Description
        :param X: input data of shape (N, D)
        :param y: vector of training labels (N, )
        :param reg: regularization strength
        """

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        N, D = X.shape

        # forward pass

        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1)
        scores = a1.dot(W2) + b2
        if y is None:
            return scores
        
        # loss calculation
        loss = 0.0

        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[np.arange(N), y])

        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        loss = data_loss + reg_loss

        # implementing backpropagation from scratch
        y = y.flatten()
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N

        dW2 = a1.T @ dscores + reg * W2
        db2 = np.sum(dscores, axis=0)

        da1 = dscores @ W2.T

        dz1 = da1.copy()
        dz1[a1 <= 0] = 0

        dW1 = X.T @ dz1 + reg * W1
        db1 = np.sum(dz1, axis=0)

        grads = {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2
        }
        
        return loss, grads
    
    def train(self, X, y, X_val, y_val, learning_rate=1e-3, 
              learning_rate_decay=0.95, reg=5e-6, num_iters=100, 
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            loss, grads = self.loss(X_batch, y_batch, reg=reg)
            loss_history.append(loss)

            for param_name in self.params:
                self.params[param_name] -= learning_rate * grads[param_name]

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            
            if it % iterations_per_epoch == 0:
                learning_rate *= learning_rate_decay
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }
    
    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
        """

        y_pred = None

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1)

        scores = a1.dot(W2) + b2

        y_pred = np.argmax(scores, axis=1)
        
        return y_pred
        