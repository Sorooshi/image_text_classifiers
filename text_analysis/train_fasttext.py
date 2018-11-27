import os
import re
import json
import string
import fasttext
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

MODEL_DIR = "FASTTEXT_MODEL_DIR"
text_data_path = "C:/Users/srsha/image_text_classifier/text_analysis/vgg_mc_sl.txt"
# text_data_path = "./toy_data.txt"

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()  #sort words by grouping inflected or variant forms of the same word.
p_stemmer = PorterStemmer()
adjectives = ['good', 'bad', 'better', 'worse', 'best', 'worst', 'small', 'medium', 'large', 'new', 'old']

def read_text(text_data_path):
    with open(text_data_path+".json", 'r') as fp:
        all_text = json.load(fp)
        text = []
        for k, v in all_text.items():
            text.append(v[1])
    return text

def clean(doc):
    adj_free = ' '.join([i for i in doc.lower().split() if i not in adjectives])
    digit_free = re.sub(r'[a-z]\d+[a-z]|[a-z]+\d+|\d+[a-z]+|\d+', ' ', adj_free)
    stop_free = " ".join([i for i in digit_free.split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    stemmed_doc = "".join(p_stemmer.stem(i) for i in normalized)
    return stemmed_doc

def create_model_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return None

create_model_dir(MODEL_DIR)

# Reading the text data
with open(text_data_path, 'r') as fp:
    text_data = [line for line in fp]

print("text_data:", text_data, len(text_data))

model_ngram = fasttext.skipgram(text_data_path, 'model_ngram')
print("skipgram:")
print(model_ngram.words)

model_bow = fasttext.cbow(text_data_path, 'model_bow')
print("BoW:")
print(model_bow)



# model_ngram_loaded = fasttext.load_model('model_ngram.bin')
# print("model_ngram.words:", len(model_ngram_loaded.words))
# print(model_ngram_loaded.words)
# print(' ')
# print(model_ngram_loaded['soroosh'], len(model_ngram_loaded['soroosh']))
#
# print(" ")
#
# model_bow_loaded = fasttext.load_model('model_bow.bin')
# print("model_bow.words:", len(model_bow_loaded.words))
# print(model_bow_loaded.words)
# print(' ')
# print(model_bow_loaded['soroosh'], len(model_bow_loaded['soroosh']))


# Data cleaning
# clean_text_data = [clean(doc).split() for doc in text_data]
# print("corpus clean:", clean_text_data, len(clean_text_data))
