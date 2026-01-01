# NumpyNet: CIFAR-10 from Scratch

A fully vectorized implementation of a 2-layer Neural Network built purely in NumPy, without using PyTorch, TensorFlow, or Keras.

This project demonstrates the core mechanics of Deep Learning, including manual backpropagation, stochastic gradient descent, and matrix vectorization.

## Performance

- **Final validation accuracy**: 57.10%
- **Test accuracy**: 55.41%

(Beating random guessing of 10% and typical linear benchmark of ~35%)

**Architecture**: Input (3072) → Hidden (100, ReLU) → Output (10, Softmax)  
**Optimization**: Stochastic Gradient Descent (SGD) with L2 Regularization  
**Loss Reduction**: 2.30 → 1.20

## Key Features

- **Fully Vectorized Code**: No slow Python loops in forward or backward passes.
- **Manual Backpropagation**: Gradients for Softmax, ReLU, and Affine layers derived and implemented from scratch.
- **Dependency-Free**: Only requires `numpy` for math and `matplotlib` for plotting. No ML frameworks used.
- **Auto-Data Pipeline**: Custom script to automatically download, extract, and preprocess CIFAR-10 binaries.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/AKMessi/neural-network-from-scratch-using-numpy.git
   cd neural-network-from-scratch
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

3. Train the model:
   ```bash
   python train.py
   ```

   This will automatically download the dataset (if missing), train the network for 3000 iterations, and output the final accuracies.

## Architecture

The network consists of a modular 2-layer architecture:

- **Input Layer**: 3072 features (32×32×3 flattened pixels)
- **Hidden Layer**: 100 neurons with ReLU activation
- **Output Layer**: 10 class scores (Softmax)

**Mathematical Flow**:

```
Z1 = X.dot(W1) + b1
A1 = np.maximum(0, Z1)          # ReLU
Scores = A1.dot(W2) + b2
Loss = Softmax_Cross_Entropy(Scores, y) + L2_Regularization
```

## Results

The model successfully overfits small batches (sanity check) and generalizes to achieve **57.10% validation** and **55.41% test** accuracy on CIFAR-10. This represents strong performance for a simple Multi-Layer Perceptron (MLP) trained on raw pixel data without data augmentation or advanced techniques.

## License

MIT License