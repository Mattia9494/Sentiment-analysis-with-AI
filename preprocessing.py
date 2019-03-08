# coding=utf-8
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer

wordnet_lemmatizer = WordNetLemmatizer()
snow_stemmer = SnowballStemmer('english')

# helper function to clean data
def process(data):
    # mantain the NOT
    #data = re.sub(r"n't", ' not', data)
    # Remove HTML special entities (e.g. &amp;)
    data = re.sub(r'\&\w*;', '', data)
    #Convert @username
    data = re.sub(r'\@\w*', '', data)
    # Remove tickers
    data = re.sub(r'\$\w*', '', data)
    # To lowercase
    data = data.lower()
    # Remove hyperlinks
    data = re.sub(r'https?:\/\/.*\/\w*', '', data)
    # Remove hashtags
    data = re.sub(r'#\w*', '', data)
    # Remove everything which is not alphanumeric
    data = re.sub(r'[^\w]', ' ', data)
    # Remove all numbers
    data = re.sub(r'[0-9]+', '', data)
    # Remove whitespace (including new line characters)
    data = re.sub(r'\s\s+', ' ', data)
    # Remove single space remaining at the front of the data.
    data = data.lstrip(' ')
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    data = ''.join(c for c in data if c <= '\uFFFF')
    # Correct words with
    data = re.sub(r'(.)\1+', r'\1\1', data)
    # Remove words with 2 or fewer letters
    data = re.sub(r'\b\w{1,2}\b', '', data)
    # Remove other elements
    data = re.sub('\n', '', data)
    data = re.sub('"b', '', data)
    data = re.sub("'b", '', data)
    return data

#stop_words = ['I', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
#           'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
#           'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
#           'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
#           'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
#           'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
#           'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
#           'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
#           'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such','own', 'same', 'so',
#           'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
#           'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
#           "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
#           "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
#           "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stop_words=['I', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def remove_stop_words(sentence):
    token_words=word_tokenize(sentence)
    sentence_wo_sw=[]
    for word in token_words:
        if word not in stop_words:
            sentence_wo_sw.append(word)
            sentence_wo_sw.append(" ")
        else:
            continue
    return "".join(sentence_wo_sw)

def lemmanouns(sentence):
    token_words=word_tokenize(sentence)
    lemma_sentence=[]
    for word in token_words:
        lemma_sentence.append(wordnet_lemmatizer.lemmatize(word, pos='n'))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)

def lemmaverbs(sentence):
    token_words=word_tokenize(sentence)
    lemma_sentence=[]
    for word in token_words:
        lemma_sentence.append(wordnet_lemmatizer.lemmatize(word, pos='v'))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)

def lemmaadjectives(sentence):
    token_words=word_tokenize(sentence)
    lemma_sentence=[]
    for word in token_words:
        lemma_sentence.append(wordnet_lemmatizer.lemmatize(word, pos='a'))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)

def stemmer(sentence):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(snow_stemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

