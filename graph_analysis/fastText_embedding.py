# https://pypi.org/project/fasttext/
# https://github.com/facebookresearch/fastText
import io
import fasttext
import fasttext.util
import sys
import os
sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# can't process n'gram
# data = load_vectors("./data/crawl-300d-2M-subword.vec")
# print(data["newyork"])
def test_load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


# can process n'gram
def load_model(fname="./data/cc.en.300.bin", is_debug=False):
    if not os.path.isfile(fname):
        path_file = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(path_file, fname)
    model = fasttext.load_model(fname)
    if is_debug:
        print(model.get_dimension())
        print(model.get_word_vector('hello').shape)
        print(model["hello"])
        # misspelled word
        print(model.get_nearest_neighbors('enviroment'))
        # nearest_neighbors
        print(model.get_nearest_neighbors('pidgey'))
        # [(0.896462, u'paris'), (0.768954, u'bourges'), (0.765569, u'louveciennes'), (0.761916, u'toulouse'), (0.760251, u'valenciennes'), (0.752747, u'montpellier'), (0.744487, u'strasbourg'), (0.74143, u'meudon'), (0.740635, u'bordeaux'), (0.736122, u'pigneaux')]
        print(model.get_analogies("berlin", "germany", "france"))
        print(model.get_analogies("psx", "sony", "nintendo"))
        # n-grams
        # [(0.790762, u'gearing'), (0.779804, u'flywheels'), (0.777859, u'flywheel'), (0.776133, u'gears'), (0.756345, u'driveshafts'), (0.755679, u'driveshaft'), (0.749998, u'daisywheel'), (0.748578, u'wheelsets'), (0.744268, u'epicycles'), (0.73986, u'gearboxes')]
        print(model.get_nearest_neighbors('gearshift'))
        import pdb; pdb.set_trace()
    print("=== Load fastText model ===")
    return model

def get_faxtText_embedding(ft_model, text):
    text = text.lower()
    print(text)
    return ft_model.get_word_vector(text)

is_debug=False
if __name__ == '__main__':
    model = load_model("./data/cc.en.300.bin", is_debug=is_debug)
    print()
