import numpy as np
import tensorflow as tf
import random as rn
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(1)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(2)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(2)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

#importing libraries
import preprocessing as pp
from timeit import default_timer as timer
from datetime import timedelta
import pandas as pd
import os
from keras.utils import to_categorical
import keras.regularizers
from keras.constraints import MinMaxNorm, UnitNorm, NonNeg, MaxNorm
from keras.initializers import Zeros, Ones, RandomNormal, RandomUniform, TruncatedNormal, VarianceScaling, Orthogonal, Identity, lecun_uniform, glorot_normal, glorot_uniform, he_normal
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, ELU, GRU, AlphaDropout, Dropout, BatchNormalization, LSTM, Activation, Embedding, PReLU, GlobalMaxPool1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib
matplotlib.use('TkAgg')

#start the times
start = timer()

# ===============
# Hyperparameters
# ===============
start_train= '2008-08-09'
end_train = '2014-12-31'
start_val = '2015-01-02'
end_val = '2016-07-01'
patience = 3
verbose = 1
Base_Dir = ''
Weights_Name = 'DJIA_weights_plus3.hdf5'
Model_Name = 'DJIA_model_plus3.h5'
Stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=verbose, mode='auto')
Checkpointer = ModelCheckpoint(filepath=os.path.join(Base_Dir, Weights_Name), verbose=verbose, save_best_only=True)
max_sequence_length = 110
vocab_size = 3000
embedding_dim = 256
hidden_layer_size = 256
dropout = 0.3
recurrent_dropout = 0.3
batch_size = 64
num_epochs = 3
l1 = 0.0001
l2 = 0.0001


#read csv file
DJIA = pd.read_csv("Combined_News_DJIA_plus3.csv", usecols=['Date', 'Label','Top1', 'Top2', 'Top3', 'Top4', 'Top5',
       'Top6', 'Top7', 'Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13',
       'Top14', 'Top15', 'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21',
       'Top22', 'Top23', 'Top24', 'Top25'])

#create training and testing dataframe on 80 % and 20 % respectively
Training_dataframe = DJIA[(DJIA['Date']>=start_train) & (DJIA['Date']<=end_train)]
Testing_dataframe = DJIA[(DJIA['Date']>=start_val) & (DJIA['Date']<=end_val)]

attrib = DJIA.columns.values

x_train = Training_dataframe.loc[:,attrib[2:len(attrib)]]
y_train = Training_dataframe.iloc[:,1]

x_test = Testing_dataframe.loc[:,attrib[2:len(attrib)]]
y_test = Testing_dataframe.iloc[:,1]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# merge the 25 news together to form a single signal
merged_x_train = x_train.apply(lambda x: ''.join(str(x.values)), axis=1)
merged_x_test = x_test.apply(lambda x: ''.join(str(x.values)), axis=1)

# ===============
# pre-process
# ===============

merged_x_train = merged_x_train.apply(lambda x: pp.process(x))
merged_x_test = merged_x_test.apply(lambda x: pp.process(x))

merged_x_train = merged_x_train.apply(lambda x: pp.remove_stop_words(x))
merged_x_test = merged_x_test.apply(lambda x: pp.remove_stop_words(x))

# tokenize and create sequences
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(merged_x_train)
x_train_sequence= tokenizer.texts_to_sequences(merged_x_train)
x_test_sequence= tokenizer.texts_to_sequences(merged_x_test)

word_index = tokenizer.word_index
input_dim = len(word_index) + 1
print('Found %s unique tokens.' % len(word_index))

x_train_sequence = pad_sequences(x_train_sequence, maxlen=max_sequence_length)
x_test_sequence = pad_sequences(x_test_sequence, maxlen=max_sequence_length)

print('Shape of training tensor:', x_train_sequence.shape)
print(x_train_sequence)
print('Shape of testing tensor:', x_test_sequence.shape)
print(x_test_sequence)

# ===============
# Model creation
# ===============
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length = max_sequence_length, mask_zero=True))
model.add(LSTM(hidden_layer_size, recurrent_dropout=recurrent_dropout, return_sequences=False))
model.add(ELU())
model.add(Dropout(dropout))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["binary_accuracy"])
model.summary()

# Fit the model and evaluate
history=model.fit(x_train_sequence, y_train, batch_size=batch_size, callbacks=[Stopping, Checkpointer],
                  validation_data=(x_test_sequence, y_test),verbose=verbose, shuffle=True, epochs=num_epochs)
score, acc = model.evaluate(x_test_sequence, y_test, batch_size=batch_size)
predict = model.predict_classes(x_test_sequence, verbose=verbose)
outputdf = pd.DataFrame({"Date":list(Testing_dataframe['Date']), "label":list(predict)})
print('Test score:', score)
print('Test accuracy:', acc)

# ===============
# Plotting graph
# ===============
#plt.subplot(211)
#plt.title("accuracy")
#plt.plot(history.history["binary_accuracy"], color="r", label="train")
#plt.plot(history.history["val_binary_accuracy"], color="b", label="validation")
#plt.legend(loc="best")
#plt.subplot(212)
#plt.title("loss")
#plt.plot(history.history["loss"], color="r", label="train")
#plt.plot(history.history["val_loss"], color="b", label="validation")
#plt.legend(loc="best")
#plt.tight_layout()
#plt.show()

# Save the model
model.save(Model_Name)

# End timer
end = timer()
print('Running time:  ' + str(timedelta(seconds=(end - start))) + '  in Hours:Minutes:Seconds')