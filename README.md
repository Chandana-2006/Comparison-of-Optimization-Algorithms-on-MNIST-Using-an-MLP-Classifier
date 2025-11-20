# 1. Import packages
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers

# 2. Load & normalize data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Form a dictionary of optimizers (removed nadam, ftrl)
optimizer_dict = {
    "adam": optimizers.Adam(learning_rate=0.001),
    "sgd": optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "rmsprop": optimizers.RMSprop(learning_rate=0.001),
    "adagrad": optimizers.Adagrad(learning_rate=0.01),
    "adadelta": optimizers.Adadelta(learning_rate=1.0)
}

# 4. Function to build MLP model
def build_mlp():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# 5. Train & store results
results = {}
histories = {}

for opt_name, opt in optimizer_dict.items():
    print(f"\nTraining with optimizer: {opt_name}")
    model = build_mlp()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=3, batch_size=128,
        validation_split=0.1, verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results[opt_name] = {"accuracy": test_acc, "loss": test_loss}
    histories[opt_name] = history.history

# 6. Display final test results
print("\nFinal Test Results:")
for opt_name, res in results.items():
    print(f"{opt_name}: Accuracy = {res['accuracy']:.4f}, Loss = {res['loss']:.4f}")

# 7. Plot accuracy & loss for each optimizer
for opt_name, history in histories.items():
    plt.figure(figsize=(10,4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.title(f"{opt_name.upper()} - Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f"{opt_name.upper()} - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
