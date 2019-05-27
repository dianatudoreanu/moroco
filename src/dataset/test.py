import numpy as np
from src.dataset.dataset_utils import DatasetUtils
from src.utils.string_utils import clean_string
from src.utils.string_utils import ALPHABET, REMOVE_CHARS, REPLACE_PAIRS, NE_CHAR, BLANK_CHAR

texts, labels = DatasetUtils.read_raw_dataset("@data/subtask1/train.txt")
texts = list(texts)
# texts = list(map(clean_string, texts))
for pair in REPLACE_PAIRS:
    for index in range(len(texts)):
        texts[index] = texts[index].replace(pair[0], pair[1])

md_texts = []
ro_texts = []
for index in range(len(texts)):
    if labels[index] == 'MD':
        md_texts.append(texts[index])
    else:
        ro_texts.append(texts[index])

def numara(texts, car, use_count=False):
    cnt = 0
    for text in texts:
        if use_count is False:
            if car in text:
                cnt += 1
        else:
            cnt += text.count(car)
    
    return cnt

scores = []
for alph in ALPHABET:
    md_score = numara(md_texts, alph, True)
    ro_score = numara(ro_texts, alph, True)

    try:
        if ro_score < md_score:
            scores.append((alph, ro_score / md_score, ro_score, md_score))
        else:
            scores.append((alph, md_score / ro_score, ro_score, md_score))
    except:
        continue

scores.sort(key = lambda x: -x[1])
alphabet = list(map(lambda x: x[0], scores))

text_join = ''.join(texts)
all_chars = list(set(text_join))

chars_list = [(k, text_join.count(k)) for k in all_chars]
chars_list.sort(key = lambda x: -x[1])

all_chars = list(map(lambda x: x[0], chars_list))

import pdb; pdb.set_trace()

texts.sort(key = lambda x: len(x))

for pair in REPLACE_PAIRS:
    for index in range(len(texts)):
        texts[index] = texts[index].replace(pair[0], pair[1])

count = 0
for text in texts:
    for ch in text:
        if ch not in ALPHABET:
            count += 1

print(count)
lengths = list(map(lambda x: len(x), texts))

texts = list(map(clean_string, texts))
print(numara(texts, BLANK_CHAR, True))

print(np.sum(lengths))
import pdb; pdb.set_trace()

