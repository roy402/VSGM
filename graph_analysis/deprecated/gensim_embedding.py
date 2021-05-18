# https://stackoverflow.com/questions/35747245/bigram-to-a-vector
# Bigram to a vector
from gensim.models import word2vec, Phrases
from glove_embedding import load_object
import gensim.downloader as api

def five():
    word_vectors = api.load("glove-wiki-gigaword-100")
    word_vectors["apple"].shape
five()
def bigram2vec(unigrams, bigram_to_search):
    bigrams = Phrases(unigrams)
    model = word2vec.Word2Vec(bigrams[unigrams])
    if bigram_to_search in model.vocab.keys():
        return model[bigram_to_search]
    else:
        return None
# docs = ['new york is is united states', 'new york is most populated city in the world','i love to stay in new york']
# token_ = [doc.split(" ") for doc in docs]
# bigram2vec(token_, "new_york")

# https://stackoverflow.com/questions/51426107/how-to-build-a-gensim-dictionary-that-includes-bigrams
def two():
    from gensim.models import Phrases
    from gensim.models.phrases import Phraser
    from gensim import models
    docs = ['new york is is united states', 'new york is most populated city in the world','i love to stay in new york']
    token_ = [doc.split(" ") for doc in docs]
    bigram = Phrases(token_, min_count=1, threshold=2, delimiter=b' ')
    bigram_phraser = Phraser(bigram)
    bigram_token = []
    for sent in token_:
        bigram_token.append(bigram_phraser[sent])
    print(bigram_token)

def three():
    docs = ['new york is is united states', 'new york is most populated city in the world','i love to stay in new york']
    sentence_stream = [doc.split(" ") for doc in docs]
    # sentence_stream=brown_raw[0:10]
    bigram = Phrases(sentence_stream, min_count=1, threshold=2, delimiter=b' ')
    trigram = Phrases(bigram[sentence_stream], min_count=1, delimiter=b' ')
    print(bigram)
    for sent in sentence_stream:
        bigrams_ = [b for b in bigram[sent] if b.count(" ") == 1]
        trigrams_ = [t for t in trigram[bigram[sent]] if t.count(" ")==2]
        print(bigrams_)
        print(trigrams_)

def four():
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format('data/GoogleGoogleNews-vectors-negative300.bin', binary=True)
    vector = model['easy']
    vector.shape
    docs = ['new york is is united states', 'new york is most populated city in the world','i love to stay in new york']
    sentence_stream = [doc.split(" ") for doc in docs]
    model["new_york"].shape
# four()

# if __name__ == '__main__':
#     objects = load_object("./data/objects.txt")
#     vector = bigram2vec(objects[0], "alarm_clock")
#     print(vector)