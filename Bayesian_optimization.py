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
from keras.initializers import Zeros, Ones, RandomNormal, RandomUniform, TruncatedNormal, VarianceScaling, Orthogonal, Identity, lecun_uniform, glorot_normal, glorot_uniform, he_uniform, he_normal
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, ELU, GRU, AlphaDropout, Dropout, BatchNormalization, LSTM, Activation, Embedding, PReLU, GlobalMaxPool1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib
matplotlib.use('TkAgg')

from hyperas import optim
from hyperas.distributions import choice, uniform, randint
from hyperopt import Trials, STATUS_OK, tpe

#start the times
start = timer()

def data():
    start_train = '2008-08-08'
    end_train = '2014-12-31'
    start_val = '2015-01-02'
    end_val = '2016-07-01'
    max_sequence_length = 110
    vocab_size = 3000
    # read csv file
    DJIA = pd.read_csv("Combined_News_DJIA.csv", usecols=['Date', 'Label', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5',
                                                          'Top6', 'Top7', 'Top8', 'Top9', 'Top10', 'Top11', 'Top12',
                                                          'Top13',
                                                          'Top14', 'Top15', 'Top16', 'Top17', 'Top18', 'Top19', 'Top20',
                                                          'Top21',
                                                          'Top22', 'Top23', 'Top24', 'Top25'])

    # create training and testing dataframe on 80 % and 20 % respectively
    Training_dataframe = DJIA[(DJIA['Date'] >= start_train) & (DJIA['Date'] <= end_train)]
    Testing_dataframe = DJIA[(DJIA['Date'] >= start_val) & (DJIA['Date'] <= end_val)]

    attrib = DJIA.columns.values

    x_train = Training_dataframe.loc[:, attrib[2:len(attrib)]]
    y_train = Training_dataframe.iloc[:, 1]

    x_test = Testing_dataframe.loc[:, attrib[2:len(attrib)]]
    y_test = Testing_dataframe.iloc[:, 1]

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

    #merged_x_train = merged_x_train.apply(lambda x: pp.lemmanouns(pp.lemmaverbs(pp.lemmaadjectives(x))))
    #merged_x_test = merged_x_test.apply(lambda x: pp.lemmanouns(pp.lemmaverbs(pp.lemmaadjectives(x))))

    #merged_x_train = merged_x_train.apply(lambda x: pp.stemmer(x))
    #merged_x_test = merged_x_test.apply(lambda x: pp.stemmer(x))

    # remove stopwords in the training and testing set
    train_without_sw = []
    test_without_sw = []
    train_temporary = list(merged_x_train)
    test_temporary = list(merged_x_test)
    s = pp.stop_words
    for i in train_temporary:
        f = i.split(' ')
        for j in f:
            if j in s:
                f.remove(j)
        s1 = ""
        for k in f:
            s1 += k + " "
        train_without_sw.append(s1)
    merged_x_train = train_without_sw

    for i in test_temporary:
        f = i.split(' ')
        for j in f:
            if j in s:
                f.remove(j)
        s1 = ""
        for k in f:
            s1 += k + " "
        test_without_sw.append(s1)
    merged_x_test = test_without_sw

    # tokenize and create sequences
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(merged_x_train)
    x_train_sequence = tokenizer.texts_to_sequences(merged_x_train)
    x_test_sequence = tokenizer.texts_to_sequences(merged_x_test)

    word_index = tokenizer.word_index
    input_dim = len(word_index) + 1
    print('Found %s unique tokens.' % len(word_index))

    x_train_sequence = pad_sequences(x_train_sequence, maxlen=max_sequence_length)
    x_test_sequence = pad_sequences(x_test_sequence, maxlen=max_sequence_length)

    print('Shape of training tensor:', x_train_sequence.shape)
    print(x_train_sequence)
    print('Shape of testing tensor:', x_test_sequence.shape)
    print(x_test_sequence)
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    return x_train_sequence, y_train, x_test_sequence, y_test


# ===============
# Model creation
# ===============

def create_model(x_train_sequence, y_train, x_test_sequence, y_test):
    verbose = 1
    max_sequence_length = 110
    vocab_size = 3000
    embedding_dim = {{choice([32, 64, 128, 256, 512])}}
    lstm = {{choice([32, 64, 128, 256, 512])}}
    num_epochs = {{choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}}
    dropout = {{uniform(0, 1)}}
    recurrent_dropout = {{uniform(0, 1)}}
    alpha = {{uniform(0, 3)}}
    batch_size = {{choice([32, 64, 128, 256])}}
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length, mask_zero=True))
    model.add(LSTM(lstm, recurrent_dropout=recurrent_dropout, return_sequences=False))
    model.add(ELU(alpha=alpha))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["binary_accuracy"])
    model.summary()

    # Fit the model and evaluate
    result = model.fit(x_train_sequence, y_train, batch_size=batch_size,
                        validation_data=(x_test_sequence, y_test), verbose=verbose, shuffle=True, epochs=num_epochs)
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history["binary_accuracy"])
    print('Best validation acc of epoch:', validation_acc)
    print('Embedding_dim: ', embedding_dim)
    print('Number of neurons: ', lstm)
    print('Epochs: ', num_epochs)
    print('Dropout: ', dropout)
    print('Recurrent Dropout: ', recurrent_dropout)
    print('Batch Size: ', batch_size)
    #print('Alpha: ', alpha)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


# End timer
end = timer()
print('Running time:  ' + str(timedelta(seconds=(end - start))) + '  in Hours:Minutes:Seconds')

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=80,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
