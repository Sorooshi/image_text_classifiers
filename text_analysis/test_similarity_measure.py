import numpy as np
from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('C:/Users/srsha/image_text_classifier/text_analysis/MODEL_DIR/vgg_mc_sl.dict')
corpus = corpora.MmCorpus('C:/Users/srsha/image_text_classifier/text_analysis/MODEL_DIR/vgg_mc_sl.mm')

Lsi_index = similarities.MatrixSimilarity.load('C:/Users/srsha/image_text_classifier/text_analysis/MODEL_DIR/vgg_mc_sl_Lsi_index.index')
Lsi_model = models.LsiModel.load("C:/Users/srsha/image_text_classifier/text_analysis/MODEL_DIR/vgg_mc_sl_Lsi_model.Lsi")

Lda_index = similarities.MatrixSimilarity.load('C:/Users/srsha/image_text_classifier/text_analysis/MODEL_DIR/vgg_mc_sl_Lda_index.index')
Lda_model = models.ldamodel.LdaModel.load("C:/Users/srsha/image_text_classifier/text_analysis/MODEL_DIR/vgg_mc_sl_Lda_model.Lda")

print(dictionary)
print(corpus)

query = "drill"
# query = "This article is about the tool. For other uses, see Hammer (disambiguation)"

query_vec_bow = dictionary.doc2bow(query.lower().split())

query_similarity_Lsi = Lsi_index[Lsi_model[query_vec_bow]]
print("query_similarity_Lsi:", np.max(query_similarity_Lsi), np.argmax(query_similarity_Lsi))


query_similarity_Lda = Lda_index[Lda_model[query_vec_bow]]
print("query_similarity_Lda:", np.max(query_similarity_Lda), np.argmax(query_similarity_Lda) )

