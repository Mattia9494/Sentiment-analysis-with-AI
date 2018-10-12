#importing libraries
import preprocessing as pp
from timeit import default_timer as timer
from datetime import timedelta
from tensorflow import set_random_seed
from numpy.random import seed
import numpy as np
import pandas as pd
import re
import os
from string import punctuation
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, ELU, GRU, Dropout, BatchNormalization, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.embeddings import Embedding
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#start the times
start = timer()
#set the seed to obtain reproducible results
seed(1)
set_random_seed(2)

# ===============
# Hyperparameters
# ===============
start_train= '2008-08-08'
end_train = '2014-12-31'
start_val = '2015-01-02'
end_val = '2016-07-01'
patience = 1
verbose = 1
Base_Dir = ''
Weights_Name = 'DJIA_weights1.hdf5'
Model_Name = 'DJIA_model1.h5'
Stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=verbose, mode='auto')
Checkpointer = ModelCheckpoint(filepath=os.path.join(Base_Dir,Weights_Name), verbose=verbose, save_best_only=True)
max_sequence_length = 400
vocab_size = 10000
embedding_dim = 128
hidden_layer_size = 256
dropout = 0.2
recurrent_dropout = 0.5
l1 = 0.01
l2 = 0.01
batch_size = 32
num_epochs = 20
learning_rate = 0.001

#read csv file
DJIA = pd.read_csv("Combined_News_DJIA.csv")

#create training and testing dataframe on 80 % and 20 % respectively
Training_dataframe = DJIA[(DJIA['Date']>=start_train) & (DJIA['Date']<=end_train)]
Testing_dataframe = DJIA[(DJIA['Date']>=start_val) & (DJIA['Date']<=end_val)]

attrib = DJIA.columns.values

x_train = Training_dataframe.loc[:,attrib[2:len(attrib)]]
y_train = Training_dataframe.iloc[:,1]

x_test = Testing_dataframe.loc[:,attrib[2:len(attrib)]]
y_test = Testing_dataframe.iloc[:,1]

# merge the 25 news together to form a single signal
merged_x_train = x_train.apply(lambda x: ''.join(str(x.values)), axis=1)
merged_x_test = x_test.apply(lambda x: ''.join(str(x.values)), axis=1)

# ===============
# pre-process
# ===============
merged_x_train = merged_x_train.apply(lambda x: pp.process(x))
merged_x_test = merged_x_test.apply(lambda x: pp.process(x))

# remove stopwords in the training and testing set
train_without_sw=[]
test_without_sw=[]
train_temporary=list(merged_x_train)
test_temporary=list(merged_x_test)
s=set(stopwords.words('english'))
for i in train_temporary:
    f=i.split(' ')
    for j in f:
        if j in s:
            f.remove(j)
    s1=""
    for k in f:
        s1+=k+" "
    train_without_sw.append(s1)
merged_x_train=train_without_sw

for i in test_temporary:
    f=i.split(' ')
    for j in f:
        if j in s:
            f.remove(j)
    s1=""
    for k in f:
        s1+=k+" "
    test_without_sw.append(s1)
merged_x_test=test_without_sw

# tokenize and create sequences
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(merged_x_train)
x_train_sequence= tokenizer.texts_to_sequences(merged_x_train)
x_test_sequence= tokenizer.texts_to_sequences(merged_x_test)
x_train_sequence = pad_sequences(x_train_sequence, maxlen=max_sequence_length)
x_test_sequence = pad_sequences(x_test_sequence, maxlen=max_sequence_length)

# ===============
# Model creation
# ===============
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length= max_sequence_length))
model.add(LSTM(hidden_layer_size, recurrent_dropout=recurrent_dropout, return_sequences=True))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(dropout))
model.add(LSTM(hidden_layer_size, recurrent_dropout=recurrent_dropout, return_sequences=False))
model.add(BatchNormalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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
plt.subplot(211)
plt.title("accuracy")
plt.plot(history.history["acc"], color="r", label="train")
plt.plot(history.history["val_acc"], color="b", label="validation")
plt.legend(loc="best")
plt.subplot(212)
plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="validation")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Save the model
model.save(Model_Name)

# End timer
end = timer()
print('Running time:  ' + str(timedelta(seconds=(end - start))) + '  in Hours:Minutes:Seconds')
