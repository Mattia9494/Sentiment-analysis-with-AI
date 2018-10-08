# coding=latin-1
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, GRU, Bidirectional, Dropout

start = timer()


#Hyperparameters
Model_Name = 'my_model.h5'
max_sequence_length = 200
validation_split = 0.25
vocab_size = 400000
embedding_dim = 300
hidden_layer_size = 256
dropout = 0.3
l1 = 0.01
l2 = 0.01
batch_size = 128
num_epochs = 20
learning_rate = 0.001

#Load data
Base_Dir = ''
GloVe_Dir = os.path.join(Base_Dir, 'glove')
Text_Data_Dir = os.path.join(Base_Dir, 'Sentiment140.csv')
df = pd.read_csv(Text_Data_Dir, encoding='latin-1', header=None)

#Organize columns
df.columns = ['sentiment', 'id', 'date', 'q', 'user', 'text']
df = df[df.sentiment != 2]
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
df = df.drop(['id', 'date', 'q', 'user'], axis=1)
df = df[['text','sentiment']]

#Preprocessing
df['text'] = df['text'].apply(pp.process)
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(df.text)
sequences = tokenizer.texts_to_sequences(df.text)
#print(sequences)
word_index = tokenizer.word_index
#print("word index =", word_index)
print('Found %s unique tokens.' % len(word_index))
preprocessed_text = pad_sequences(sequences, maxlen=max_sequence_length)
labels = df.sentiment
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', preprocessed_text.shape)
print('Shape of label tensor:', labels.shape)

# Split the data into training and validation set
x_train, x_val, y_train, y_val = train_test_split(preprocessed_text, labels, test_size=validation_split, shuffle=False)

# Normalize
print('Normalizing...')
scaler = MinMaxScaler()
x_train = scaler.fit_transform(np.array(x_train))
x_val = scaler.fit_transform(np.array(x_val))
print('Normalization completed!')

# Implement GloVe
embeddings_index = {}
f = open(os.path.join(GloVe_Dir, 'glove.6B.{}d.txt'.format(embedding_dim)))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Build the model
print('Build model...')
model = Sequential()
# Load pre-trained word embeddings into an Embedding layer
# Note: we set "trainable = False" so as to keep the embeddings fixed
model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                    input_length=max_sequence_length, trainable=False))
model.add(Bidirectional(GRU(hidden_layer_size, kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2),
                            kernel_initializer='uniform', dropout=dropout, recurrent_dropout=dropout,
                            return_sequences=True, activation='relu')))
model.add(Dropout(dropout))
model.add(Bidirectional(GRU(hidden_layer_size, kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2),
                            kernel_initializer='uniform', dropout=dropout, recurrent_dropout=dropout,
                            return_sequences=False, activation='relu')))
model.add(Dropout(dropout))
model.add(Dense(2, activation='softmax'))

# Optimizer
adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# Compile my model
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

print('Train...')

# Callbacks
Stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')
Checkpointer = ModelCheckpoint(filepath=os.path.join(Base_Dir,'weights.hdf5'), verbose=1, save_best_only=True)

# Fit the model
history = model.fit(x_train, y_train, batch_size=batch_size, callbacks=[Stopping, Checkpointer],
                    epochs=num_epochs, validation_data=(x_val, y_val))
score, acc = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=1)

# Create a summary, a plot and print the scores of the model
model.summary()

print('Test score:', score)
print('Test accuracy:', acc)

# plot loss function
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

model.save(Model_Name)

end = timer()
print('Running time:  ' + str(timedelta(seconds=(end - start))) + '  in Hours:Minutes:Seconds')
