from keras.models import load_model
import DJIA_sentiment as djia
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

model = load_model('DJIA_model.h5')

news_aggregate = ['Salvini reveals list of demands to Brussels as he threatens to veto EU plans '
                  'ITALYs Deputy Prime Minister Matteo Salvini warned he will exercise his right to'
                  'veto EU decisions unless Brussels agrees to compromise on a list of key demands.'
                  'Speaking to Aljazeera, the eurosceptic Interior Minister pledged to convince Brussels'
                  'to cave into his demands on immigration, economic sanctions on Russia and future trade agreements.'
                  'Mr Salvini warned Italy will be prepared to veto any future decisions that will not be']
                  
                  
#vectorizing the news from whch we want to have a signal by the pre-fitted tokenizer instance
news_aggregate = djia.tokenizer.texts_to_sequences(news_aggregate)

#padding the string to have exactly the same shape as the model input
news_aggregate = pad_sequences(news_aggregate, maxlen=djia.max_sequence_length, dtype='object')

print(news_aggregate)
sentiment = model.predict(news_aggregate,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative sentiment")
elif (np.argmax(sentiment) == 1):
    print("positive sentiment")
