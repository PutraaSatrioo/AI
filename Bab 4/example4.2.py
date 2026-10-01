# Example 4.2 Multiple Layer Perceptron

from sklearn.neural_network import MLPClassifier

# Input data
X = [
    [0., 0.],
    [1., 1.],
    [0., 1.],
    [1., 0.]
]

# Target output
y = [0, 1, 1, 1]

# Create MLP model
clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1
)

# Train the model
clf.fit(X, y)

# Prediction
print(clf.predict([
    [2., 2.],
    [-1., -2.]
]))

# Display weight matrix shapes
print([coef.shape for coef in clf.coefs_])