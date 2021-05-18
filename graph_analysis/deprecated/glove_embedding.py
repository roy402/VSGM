import numpy as np
import h5py
import json
import re
from gensim.models import word2vec, Phrases


def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File, 'r', encoding='utf-8')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel), " words loaded!")
    return gloveModel


def dump_json(data, path):
    with open(path, "w") as outfile:
        json.dump(data, outfile)


def load_object(File):
    objects = open(File).readlines()
    objects = [o.strip() for o in objects]
    for i in range(len(objects)):
        o = re.findall('[A-Z][^A-Z]*', objects[i])
        o = "_".join(o).lower()
        objects[i] = o
    return objects


def search_object(glove, objects):
    find = {}
    cantf = {}
    for o in objects:
        try:
            find[o.lower()] = glove[o.lower()]
        except Exception as e:
            cantf[o.lower()] = None
    return find, cantf


def bigram2vec(unigrams, bigram_to_search):
    bigrams = Phrases(unigrams)
    model = word2vec.Word2Vec(bigrams[unigrams])
    if bigram_to_search in model.vocab.keys():
        return model[bigram_to_search]
    else:
        return None


# save h5  & cantf json
def save_glove_file(find, cantf):
    glove_file = './data/glove_300d_{}.h5'.format(len(list(find.values())))
    with h5py.File(glove_file, 'w') as hf:
        for k, v in find.items():
            hf.create_dataset(k, data=v)

    find_path = "./data/find.json"
    cantf_path = "./data/cantf.json"
    import pdb; pdb.set_trace()
    dump_json(find, find_path)
    dump_json(cantf, cantf_path)


def get_glove_embedding():
    # load glove and object
    gloveModel = loadGloveModel("./data/glove.42B.300d.txt")
    objects = load_object("./data/objects.txt")
    find, cantf = search_object(gloveModel, objects)
    save_glove_file(find, cantf)

if __name__ == '__main__':
    objects = load_object("./data/objects.txt")
    get_glove_embedding()