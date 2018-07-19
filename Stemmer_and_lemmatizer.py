#Example of some use simple cases of the NLTK stemmer and lemmatizer for NLP

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
test_word = 'fighting'
word_stem = stemmer.stem(test_word)
word_lemma= lemmatizer.lemmatize(test_word)
word_lemma_verb = lemmatizer.lemmatize(test_word, pos = 'v')
word_lemma_adj = lemmatizer.lemmatize(test_word, pos = 'a')
word_lemma_noun = lemmatizer.lemmatize(test_word, pos = 'n')
print(word_stem, word_lemma, word_lemma_verb, word_lemma_adj, word_lemma_noun)

#print all the available languages that can be used to create subclasses
print(SnowballStemmer.languages)

#find the definition of a word and make an example
syn = wordnet.synsets('pain')
print(syn[0].definition())
print(syn[0].examples())

#get synonymous words
synonyms = []
for syn in wordnet.synsets('sad'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
        print(synonyms)

#get antonyms words
antonyms = []
for syn in wordnet.synsets('happy'):
    for l in syn.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
            print(antonyms)
