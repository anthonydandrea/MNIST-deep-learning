
# coding: utf-8

# In[191]:


import numpy as np
np.random.seed(123)


# In[192]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten


# In[193]:


from keras.layers import Conv2D, MaxPooling2D


# In[194]:


from keras.utils import np_utils


# In[195]:


from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[196]:


print(X_train.shape)


# In[197]:


get_ipython().magic('matplotlib inline')

from matplotlib import pyplot as plt
plt.imshow(X_train[0])
print(X_train.shape[0])


# In[198]:


from keras import backend
backend.set_image_dim_ordering('th')


# In[199]:


X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test = X_test.reshape(X_test.shape[0],1,28,28)


# In[200]:


print(X_train.shape)


# In[201]:


print(type(X_train[0][0][0][0]))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(type(X_train[0][0][0][0]))
X_train /= 255
X_test /= 255


# In[202]:


print(y_train.shape)


# In[203]:


print(y_train[:10])


# In[204]:


Y_train = np_utils.to_categorical(y_train,10)
Y_test= np_utils.to_categorical(y_test,10)


# In[205]:


print(Y_train.shape)
print(Y_train[:10])


# In[206]:


print(Y_train[:10])


# In[207]:


model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(1,28,28)))


# In[208]:


print(model.output_shape)


# In[209]:


model.add(Convolution2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


# In[210]:


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[211]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

print(model.summary())


# In[212]:


model.fit(X_train[:10000],Y_train[:10000],
         batch_size=100, epochs=2, verbose=1)


# In[213]:


score = model.evaluate(X_test,Y_test,verbose=0)


# In[214]:


score

