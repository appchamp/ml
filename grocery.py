import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.optimizers import RMSprop

#far too much repetition, just cleaning the data for the neural net
df = pd.read_csv('small_train.csv')

#cleanup
#df['newdate'] = df['date'] - df['date'].min()
df['newdate'] = (pd.to_datetime(df['date']) - pd.to_datetime(df['date'].min())) / np.timedelta64(1, 'D')
df['onpromotion'] = df['onpromotion'].map({'False':0, 'True':1})
df = df.drop(['id', 'onpromotion', 'date'], axis=1)

#split
train, test = train_test_split(df, test_size=0.2)
#train = train.dropna()

X_train = train.drop(['unit_sales'], 1)
X_train = X_train.as_matrix()

X_test = test.drop(['unit_sales'], 1)
X_test = X_test.as_matrix()

y_train = train['unit_sales']
y_test = test['unit_sales']

# nb_classes = 2
nb_epoch = 100
batch_size = 4000

model = Sequential()
model.add(Dense(100, activation="relu", input_shape = X_train.shape[1:]))
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense( 1))
model.summary()

#model.compile(loss='mean_squared_error', optimizer='sgd')
#model.compile(loss='mean_squared_error', optimizer=RMSprop())
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model.fit(X_train, y_train, batch_size = batch_size, epochs=nb_epoch, validation_data=(X_test, y_test), verbose=2)


y_test_new = model.predict(X_test, batch_size=32)