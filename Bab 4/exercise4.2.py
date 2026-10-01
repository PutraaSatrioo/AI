# Example 4.2 - Multiple Layer Perceptron with 3 Inputs

from sklearn.neural_network import MLPClassifier

# Dataset with three inputs
X = [
    [0., 0., 0.],
    [1., 1., 1.],
    [0., 0., 1.],
    [1., 0., 0.],
    [0., 1., 0.],
    [1., 1., 0.],
    [1., 0., 1.],
    [0., 1., 1.]
]

# Target output
y = [0, 1, 1, 1, 1, 1, 1, 1]

# Create the MLP classifier
clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1
)

# Train the model
clf.fit(X, y)

# Test the model with three inputs
print(clf.predict([
    [1., 1., 1.],
    [0., 0., 0.]
]))

# Display the shape of the weight matrices
print([coef.shape for coef in clf.coefs_])