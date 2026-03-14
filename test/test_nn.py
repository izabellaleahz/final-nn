import numpy as np
import pytest
from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs

# helper to make a small network for testing
def _make_nn(arch=None, loss='binary_cross_entropy', lr=0.01, seed=42):
    if arch is None:
        arch = [
            {'input_dim': 2, 'output_dim': 3, 'activation': 'relu'},
            {'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'}
        ]
    return NeuralNetwork(arch, lr=lr, seed=seed, batch_size=4, epochs=10, loss_function=loss)


def test_single_forward():
    nn = _make_nn()
    W = np.array([[0.5, -0.3], [0.2, 0.8]])
    b = np.array([[0.1], [-0.1]])
    A_prev = np.array([[1.0], [2.0]])

    A_curr, Z_curr = nn._single_forward(W, b, A_prev, 'relu')

    # Z = W @ A + b = [[0.5-0.6+0.1], [0.2+1.6-0.1]] = [[0.0], [1.7]]
    assert Z_curr.shape == (2, 1)
    np.testing.assert_allclose(Z_curr, [[0.0], [1.7]], atol=1e-6)
    # relu: max(0, Z)
    np.testing.assert_allclose(A_curr, [[0.0], [1.7]], atol=1e-6)


def test_forward():
    nn = _make_nn()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 samples, 2 features

    output, cache = nn.forward(X)

    # output shape should be (output_dim=1, batch_size=2)
    assert output.shape == (1, 2)
    # cache should have A0, A1, A2, Z1, Z2
    assert 'A0' in cache
    assert 'Z1' in cache
    assert 'Z2' in cache
    # all outputs should be between 0 and 1 (sigmoid at last layer)
    assert np.all(output >= 0) and np.all(output <= 1)


def test_single_backprop():
    nn = _make_nn()
    np.random.seed(42)
    W = np.random.randn(3, 2) * 0.1
    b = np.random.randn(3, 1) * 0.1
    A_prev = np.random.randn(2, 4)
    Z_curr = np.random.randn(3, 4)
    dA_curr = np.random.randn(3, 4)

    dA_prev, dW, db = nn._single_backprop(W, b, Z_curr, A_prev, dA_curr, 'relu')

    # check shapes
    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape


def test_predict():
    nn = _make_nn()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    pred = nn.predict(X)

    assert pred.shape == (1, 2)
    # sigmoid output: values between 0 and 1
    assert np.all(pred >= 0) and np.all(pred <= 1)


def test_binary_cross_entropy():
    nn = _make_nn()
    y = np.array([[1, 0, 1]])
    y_hat = np.array([[0.9, 0.1, 0.8]])

    loss = nn._binary_cross_entropy(y, y_hat)

    assert isinstance(loss, float)
    assert loss > 0

    # perfect prediction should give near-zero loss
    perfect_loss = nn._binary_cross_entropy(y, np.array([[0.9999, 0.0001, 0.9999]]))
    assert perfect_loss < loss


def test_binary_cross_entropy_backprop():
    nn = _make_nn()
    y = np.array([[1, 0, 1]])
    y_hat = np.array([[0.9, 0.1, 0.8]])

    dA = nn._binary_cross_entropy_backprop(y, y_hat)

    assert dA.shape == y.shape


def test_mean_squared_error():
    nn = _make_nn(loss='mean_squared_error')
    y = np.array([[1.0, 0.0, 0.5]])
    y_hat = np.array([[0.9, 0.1, 0.6]])

    loss = nn._mean_squared_error(y, y_hat)

    assert isinstance(loss, float)
    assert loss > 0

    # zero error should give zero loss
    zero_loss = nn._mean_squared_error(y, y)
    assert zero_loss == 0.0


def test_mean_squared_error_backprop():
    nn = _make_nn(loss='mean_squared_error')
    y = np.array([[1.0, 0.0]])
    y_hat = np.array([[0.8, 0.2]])

    dA = nn._mean_squared_error_backprop(y, y_hat)

    assert dA.shape == y.shape
    # gradient should point from y_hat toward y
    np.testing.assert_allclose(dA, y_hat - y, atol=1e-6)


def test_sample_seqs():
    seqs = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']
    labels = [True, False, False, False, False]

    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    # should be balanced
    assert sampled_labels.count(True) == sampled_labels.count(False)
    # total should be 2x the majority class
    assert len(sampled_seqs) == 8  # 4 neg + 4 pos (oversampled)


def test_one_hot_encode_seqs():
    seqs = ['AT', 'GC']
    encoded = one_hot_encode_seqs(seqs)

    assert encoded.shape == (2, 8)  # 2 seqs, each 2 bases * 4 = 8
    # AT -> [1,0,0,0, 0,1,0,0]
    np.testing.assert_array_equal(encoded[0], [1, 0, 0, 0, 0, 1, 0, 0])
    # GC -> [0,0,0,1, 0,0,1,0]
    np.testing.assert_array_equal(encoded[1], [0, 0, 0, 1, 0, 0, 1, 0])
