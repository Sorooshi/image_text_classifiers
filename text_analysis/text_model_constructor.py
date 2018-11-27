import os
import re
import json
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models, similarities

NUM_CLASSES = NUM_TOPICS = 8
MODEL_DIR = "MODEL_DIR"
text_data_path = "C:/Users/srsha/image_text_classifier/img_downloader/vgg_mc_sl"

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()  #sort words by grouping inflected or variant forms of the same word.
p_stemmer = PorterStemmer()
adjectives = ['good', 'bad', 'better', 'worse', 'best', 'worst', 'small', 'medium', 'large', 'new', 'old']

def read_text(text_data_path):
    # pages_contents = pd.read_csv(text_data_path + ".csv")
    # text = pages_contents['PAGE TITLE'].tolist()
    # corpus_topic = pages_contents['topic']
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
text_data = read_text(text_data_path)
print("text_data:", text_data, len(text_data))

# Data cleaning
clean_text_data = [clean(doc).split() for doc in text_data]
print("corpus clean:", clean_text_data, len(clean_text_data))

# Tokenizing (assigning unique integer index)
dictionary = corpora.Dictionary(clean_text_data)
dictionary.save(MODEL_DIR + '/vgg_mc_sl.dict')

# Bag of Words
corpus = [dictionary.doc2bow(doc) for doc in clean_text_data]
corpora.MmCorpus.serialize(MODEL_DIR + '/vgg_mc_sl.mm', corpus)  # store to disk, for later use

# Tfidf
Tfidf = models.TfidfModel(corpus)
Tfidf_corpus = Tfidf[corpus]

# LSI:
Lsi = models.LdaModel
Lsi_model = Lsi(corpus, id2word=dictionary, ) #num_topics=NUM_TOPICS
Lsi_model.save(MODEL_DIR+'/vgg_mc_sl_Lsi_model.Lsi')

Lsi_index = similarities.MatrixSimilarity(Lsi_model[corpus])
Lsi_index.save(MODEL_DIR+'/vgg_mc_sl_Lsi_index.index')

print(" ")

# LDA:
Lda = models.ldamodel.LdaModel # initialize a python object.
Lda_model = Lda(corpus, id2word=dictionary, ) #num_topics=NUM_TOPICS
Lda_model.save(MODEL_DIR+'/vgg_mc_sl_Lda_model.Lda')

Lda_index = similarities.MatrixSimilarity(Lsi_model[corpus])
Lda_index.save(MODEL_DIR+'/vgg_mc_sl_Lda_index.index')


#Random Projection (Rp):
Rp = models.RpModel
Rp_model = Rp(Tfidf_corpus,)


# This section is a dirty implementation for testing the similarity measurement of a new query.
# The main testing code can be find in test_similarity_measure.py

print(" ")
# query = "A modern day hammer is a tool consisting of a weighted " \
#         "head fixed to a long handle that is swung to deliver an impact to a small area of an object."

query = "Research Group Methods for analysis and visualisation of web corpora"

query_vec_bow = dictionary.doc2bow(query.lower().split())
Lsi_sims = Lsi_model[query_vec_bow]
Lda_sims = Lda_model[query_vec_bow]
Rp_sims = Rp_model[query_vec_bow]
# Rp_sims_sorted = sorted(enumerate(Rp_sims), key=lambda item:item[1])

print("Lsi sim:", Lsi_sims)
print(" ")
print("Lds sim:", Lda_sims)
print(" ")
print("Rp sim:", Rp_sims)
