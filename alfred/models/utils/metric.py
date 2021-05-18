import re
import string
import collections


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class AccuracyMetric():
    def __init__(self):
        self.reset()

    def reset(self):
        self.store = {}

    def write_summary(self, summary_writer, n_iter, loss_name="train/precision"):
        for k, v in self.store.items():
            precision = self.get_precision(k)
            summary_writer.add_scalar(loss_name + "_" + k, precision, n_iter)

    def get_precision(self, k):
        return self.store[k]["corrects"] / self.store[k]["num_samples"]

    def __call__(self, name, label, predict):
        if name not in self.store:
            self.store[name] = {
                "corrects": 0.,
                "num_samples": 0.
            }
        predict = predict.clone().detach().max(1)[1].view(-1)
        correct = predict.eq(label).view(-1).float().sum(0)
        num_samples = len(label)
        self.store[name]["corrects"] += correct
        self.store[name]["num_samples"] += num_samples