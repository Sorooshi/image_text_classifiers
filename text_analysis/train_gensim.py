import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models, similarities


NUM_TOPICS = 2
NUM_WORDS = 3
documents = ["Graph Theory can be considered as an old branch of computer science and new branch of Mathematics 125.",
             "Graph Theory also plays an important role in Network analysis.",
             "Professor Stanley Wassermann is one the most important"
             " researchers in the field of network analysis and Graph Theory.",

             "01Soroosh Shallieh is a 0PhD1 student at NRU-HSE.", "He comes from Iran and he studies "
                                                              "and works in Russia.",
             "Soroosh's supervisor at university is Professor Boris00 Mirkin who is an outstanding supervisor 0z z2 x0z."]

def clean(doc):
    adj_free = ' '.join([i for i in doc.lower().split() if i not in adjectives])
    digit_free = re.sub(r'[a-z]\d+[a-z]|[a-z]+\d+|\d+[a-z]+|\d+', ' ', adj_free)
    stop_free = " ".join([i for i in digit_free.split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    stemmed = "".join(p_stemmer.stem(i) for i in normalized)
    print(stemmed)
    return stemmed

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()  # sort words by grouping inflected or variant forms of the same word.
p_stemmer = PorterStemmer()
adjectives = ['good', 'bad', 'better', 'worse', 'best', 'worst', 'small', 'medium', 'large', 'new', 'old']

clean_text = [clean(doc).split() for doc in documents]

# assign a integer Id to each term appeared in the corpus
dictionary = corpora.Dictionary(clean_text)
dictionary.save('soroosh.dict')
# dictionary = corpora.Dictionary.load("soroosh.dict")

# Bag of Word
corpus = [dictionary.doc2bow(doc) for doc in clean_text]
corpora.MmCorpus.serialize("soroosh.mm", corpus)

# Term frequency * Inverse Document Frequency
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
for i in corpus_tfidf:
    print("thidf:", max(i))
print(" ")

# LDA dimensionality Reduction from Tfidf/BoG to Latent Space (Transform)
Lsi = models.LsiModel
Lsi_model = Lsi(corpus_tfidf, id2word=dictionary ,num_topics=NUM_TOPICS,)

for i in Lsi_model.print_topics(num_topics=NUM_TOPICS, num_words=NUM_WORDS):
    print("lsi:", i)
print(" ")

# LDA dimensionality Reduction from BoG to Topic Space (Transform)
Lda = models.ldamodel.LdaModel
Lda_model = Lda(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
for i in Lda_model.print_topics(num_topics=NUM_TOPICS, num_words=NUM_WORDS):
    print("LDA topic modeling:", i)
print(" ")

# Reduce Vector space dimensionality. Approximating Tfidf distance between two document (Random Projection)
Rp = models.RpModel
Rp_model = Rp(corpus_tfidf, num_topics=2)
print(Rp_model)
print(" ")

# Hierarchical Dirichlet Process, A Bayesian non-parametric method
Hdp = models.HdpModel
Hdp_model = Hdp(corpus, id2word=dictionary)

for i in Hdp_model.print_topics(num_topics=2, num_words=3):
    print("Hdp:", i)