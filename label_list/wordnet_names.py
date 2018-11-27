import json
from nltk.corpus import wordnet as wn


# wordnet_lemmas = [word for word in wn.all_lemma_names()]
# with open("wordnet_lemmas.txt", 'w') as fp:
#     fp.write('\n'.join('{}'.format(word) for word in wordnet_lemmas))

wordnet_sysnset = [synset for synset in list(wn.all_synsets('n'))]
with open("wordnet_synsets.txt", 'w') as fp:
    fp.write('\n'.join('{}'.format(synset) for synset in wordnet_sysnset))


''' 
wordnet_lemmas = {}
k = 0
for word in wn.all_lemma_names():
    wordnet_lemmas[k] = word
    k +=1

print(len(wordnet_lemmas))

with open("wordnet_lemmas.json", 'w') as fp:
    json.dump(wordnet_lemmas, fp)


wordnet_synsets = {}
kk = 0
for synset in list(wn.all_synsets('n')):
    wordnet_synsets[kk] = str(synset)
    kk += 1

print(len(wordnet_synsets))

with open("wordnet_synsets.json", 'w') as fp:
    json.dump(wordnet_synsets, fp)
#

'''
