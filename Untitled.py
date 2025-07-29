#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# In[3]:


#Load IMDB dataset

max_features = 2000
maxlen = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)


# In[4]:


#input length

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


# In[5]:


#LSTM model
model = keras.Sequential([
    layers.Embedding(input_dim=max_features, output_dim=32, input_length=maxlen),
    layers.LSTM(32, return_sequences=False),
    layers.Dense(1, activation='sigmoid')
])


# In[6]:


#Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[9]:


#Train model
history = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)


# In[10]:


#Evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)


# In[12]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy during training')
plt.show()


# In[ ]:




