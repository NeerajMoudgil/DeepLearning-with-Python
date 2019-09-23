# IMDB dataset: a set of 50,000 highly polarized reviews from the Internet Movie Database
from tensorflow.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
num_words=10000)
# The argument num_words=10000 means you’ll only keep the top 10,000 most frequently occurring words in the training data. Rare words will be discarded. This allows
# you to work with vector data of manageable size.

print(train_data.shape)
print(train_labels.shape)

print(train_data[0])
print(train_labels[0])
print(test_data[0])
print(len(train_data))
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

import numpy as np

text = "very good movie loved it"
list_text= text.split(" ")
arr=[]
for t in list_text:
    for (key, value) in word_index.items():
        if key ==t:
            arr.append(value)

np_arr = np.empty(shape=(1,), dtype=object)
np_arr[0]=arr

print(np_arr.shape)

print(np_arr[0])
print(type(np_arr[0]))

print(train_data.shape)
print("type===",type(train_data[0]))
print(np_arr)
print(len(train_data))
print(len(np_arr))


encoded_review= word_index.items()

print(decoded_review)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    print(results.shape)
    print(results.ndim)

    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
testing_rev= vectorize_sequences(np_arr)

print(x_train[0])

x_test = vectorize_sequences(test_data)

print(x_test[0])
print(testing_rev)
print(x_test[0].shape)
print(testing_rev.shape)


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(y_train)

#  Model

from tensorflow.keras import models
from tensorflow.keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])

history = model.fit(partial_x_train,
partial_y_train,
epochs=20,
batch_size=512,
validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test)
predicted = model.predict(testing_rev)
print(predicted)
print(results)

