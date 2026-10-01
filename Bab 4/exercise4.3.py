import time

# Record the training start time
start_time = time.time()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("Model:")
model.summary()

# Train the model
model.fit(
    X_train,
    y_train,
    batch_size=50,
    validation_split=0.2,
    epochs=100,
    verbose=1
)

# Calculate training time
training_time = time.time() - start_time

# Evaluate the model
results = model.evaluate(X_test, y_test)

print('Training time:', training_time, 'seconds')
print('Loss:', results[0])
print('Accuracy:', results[1])