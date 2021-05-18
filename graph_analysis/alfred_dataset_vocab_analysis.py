import os
import sys
import torch
import numpy as np
import h5py
import json
import re
from fastText_embedding import load_model, get_faxtText_embedding
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
path = os.path.dirname(os.path.abspath(__file__))


def get_alfred_dataset_used_vocab():
    vocab = torch.load("./data/pp.vocab")
    action_low_word = vocab['action_low'].index2word(list(range(0, len(vocab['action_low']))))
    action_high_word = vocab['action_high'].index2word(list(range(0, len(vocab['action_high']))))
    word_word = vocab['word'].index2word(list(range(0, len(vocab['word']))))
    print("=== Get pp.vocab ===")
    print("action_low_word: \n", action_low_word)
    print("action_high_word: \n", action_high_word)
    print("word_word: \n", word_word)
    return action_low_word, action_high_word, word_word


def load_object(File):
    objects = open(File).readlines()
    objects = [o.strip() for o in objects]
    return objects


def create_graph_word_embedding(ft_model, objects):
    word_embedding = {}
    for o in objects:
        vector = get_faxtText_embedding(ft_model, o)
        word_embedding[o] = vector.tolist()
    # import pdb; pdb.set_trace()
    file_path = 'data/fastText_300d_{}'.format(len(list(word_embedding.values())))
    file_path = os.path.join(path, file_path)
    _save_h5py_file(word_embedding, file_path)
    _dump_json(word_embedding, file_path)


# save h5  & cantf json
def _save_h5py_file(data, path):
    with h5py.File(path + ".hdf5", 'w') as hf:
        for k, v in data.items():
            hf.create_dataset(k, data=v)


def _dump_json(data, path):
    with open(path + ".json", "w") as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    objects = load_object("./data/objects.txt")
    print(objects)
    # action_low_word, action_high_word, word_word = get_alfred_dataset_used_vocab()
    ft_model = load_model("./data/cc.en.300.bin", is_debug=False)
    create_graph_word_embedding(ft_model, objects)