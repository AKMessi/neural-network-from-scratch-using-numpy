import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_cifar10
from neural_net import TwoLayerNet

def run_training():
    # load data
    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = load_cifar10()

    mask = range(49000, 50000)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(49000)
    X_train = X_train[mask]
    y_train = y_train[mask]

    print(f"Train data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # setup model parameters
    input_size = 32 * 32 * 3
    hidden_size = 100
    num_classes = 10

    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # train
    print("\n Starting training...")

    stats = net.train(X_train, y_train, X_val, y_val,
                      num_iters=3000,
                      batch_size=200,
                      learning_rate=1e-3,
                      learning_rate_decay=0.95,
                      reg=0.25,
                      verbose=True)
    
    # final results
    val_acc = stats['val_acc_history'][-1]
    print(f"\n Final validation accuracy: {val_acc * 100:.2f}%")

    # sanity check
    print(f"\n Evaluating on test set...")
    test_acc = (net.predict(X_test) == y_test).mean()
    print(f"\n Test accuracy: {test_acc * 100:.2f}%")

    plt.plot(stats['loss_history'])
    plt.title('Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    run_training()