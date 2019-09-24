from tensorflow.keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
print(train_data[0])
print(train_targets)

# It would be problematic to feed into a neural network values that all take wildly different ranges.
# The network might be able to automatically adapt to such heterogeneous
# data, but it would definitely make learning more difficult

# feature-wise normalization

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

from tensorflow.keras import models
from tensorflow.keras import layers


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
    input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# k-cross validation
# It consists of splitting the available data into K partitions (typically K = 4 or 5),
# instantiating K identical models, and training each one on K – 1 partitions while evaluating on
# the remaining partition. The validation score for the model used is then the average of
# the K validation scores obtained

import numpy as np
k=4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
all_mae_histories = []


for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
    [train_data[:i * num_val_samples],
    train_data[(i + 1) * num_val_samples:]],
    axis=0)
    partial_train_targets = np.concatenate(
    [train_targets[:i * num_val_samples],
    train_targets[(i + 1) * num_val_samples:]],
    axis=0)
    model = build_model()
    # model.fit(partial_train_data, partial_train_targets,
    #           epochs=num_epochs, batch_size=1, verbose=0)
    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    # all_scores.append(val_mae)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    average_mae_history = [
        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
             previous = smoothed_points[-1]
             smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
        return smoothed_points

average_mae_history = [
np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# According to this plot, validation MAE stops improving significantly after 80 epochs.
# Past that point, you start overfitting so we use eopchs=80

model = build_model()
model.fit(train_data, train_targets,
epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(test_mse_score)
print(test_mae_score)