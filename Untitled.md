```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
```


```python
#Load IMDB dataset

max_features = 2000
maxlen = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
    [1m17464789/17464789[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 0us/step



```python
#input length

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
```


```python
#LSTM model
model = keras.Sequential([
    layers.Embedding(input_dim=max_features, output_dim=32, input_length=maxlen),
    layers.LSTM(32, return_sequences=False),
    layers.Dense(1, activation='sigmoid')
])
```

    /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/keras/src/layers/core/embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.
      warnings.warn(



```python
#Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```


```python
#Train model
history = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)
```

    Epoch 1/3
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 35ms/step - accuracy: 0.8862 - loss: 0.2761 - val_accuracy: 0.8630 - val_loss: 0.3266
    Epoch 2/3
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 35ms/step - accuracy: 0.8871 - loss: 0.2709 - val_accuracy: 0.8710 - val_loss: 0.3203
    Epoch 3/3
    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 34ms/step - accuracy: 0.8978 - loss: 0.2504 - val_accuracy: 0.8698 - val_loss: 0.3170



```python
#Evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)
```

    [1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 6ms/step - accuracy: 0.8652 - loss: 0.3154
    Test loss: 0.31147411465644836
    Test accuracy: 0.8670399785041809



```python
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy during training')
plt.show()
```


    
![png](output_7_0.png)
    



```python

```
