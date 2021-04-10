import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

df = pd.read_csv("data/fft-data.csv").dropna().reset_index(drop=True)

x = df.drop(columns=['Time', 'Label']).to_numpy(dtype='float64')
y = df['Label'].to_numpy(dtype='float64')

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=False)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(200, input_dim=100, activation='softmax'),
    tf.keras.layers.Dense(100, activation='softmax'),
    tf.keras.layers.Dense(70, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])

model_out = model.fit(x_train, y_train, epochs=800, validation_data=[x_test, y_test])

print("Training accuracy: {:.5f}".format(np.mean(model_out.history['accuracy'])))
print("Validation accuracy: {:.5f}".format(np.mean(model_out.history['val_accuracy'])))

y_prediction = model.predict(x_test)
print(y_test, y_prediction)
