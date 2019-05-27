import numpy as np
import re
from multiprocessing import cpu_count, Pool
from functools import partial

def check_multi(indices, texts):
    print(indices)
    result = []
    for index1 in range(indices[0], indices[1]):
        result_row = []
        for index2 in range(indices[2], indices[3]):
            s1 = texts[index1][:-3]
            s2 = texts[index2][:-3]

            s1 = s1.lower()
            s2 = s2.lower()

            h = {}
            h2 = {}

            for i in range(len(s2) - 5):
                h2[s2[i:i+6]] = True

            count = 0
            for i in range(len(s1) - 5):
                if s1[i:i+6] in h2 and s1[i:i+6] not in h:
                    count += 1
                    h[s1[i:i+6]] = True
            result_row.append(count)
        result.append(result_row)

    return result


def check(s1, s2):
    s1 = s1[:-3]
    s2 = s2[:-3]
    # s1 = s1.replace('$NE$', '#')
    # s2 = s2.replace('$NE$', '#')

    s1 = s1.lower()
    s2 = s2.lower()

    # print(s1)
    # print(s2)

    h = {}
    h2 = {}

    for i in range(len(s2) - 5):
        h2[s2[i:i+6]] = True

    result = 0
    for i in range(len(s1) - 5):
        if s1[i:i+6] in h2 and s1[i:i+6] not in h:
            result += 1
            h[s1[i:i+6]] = True

    return result


kernel = np.load('../data/K6.npy')

texts = open('../data/subtask1/train.txt').read()
texts = texts.split('\n')[:-1]

ids = list(map(int, open('../data/train_ids.txt').read().split('\n')[:-1]))

import time

print(len(texts), kernel.shape)

partial_method = partial(check_multi, texts=texts)
pool = Pool(processes=cpu_count())

all_indices = []
lin_space = list(map(int, np.linspace(0, len(texts), cpu_count() + 1).tolist()))

for index in range(cpu_count()):
    all_indices.append([lin_space[index], lin_space[index + 1], 0, 100])

last = time.time() * 1000

result = pool.map(partial_method, all_indices)

pool.close()
pool.join()

now = time.time() * 1000
print(now - last, 'ms')


import pdb; pdb.set_trace()
last = now

for i in range(len(texts)):
    now = time.time() * 1000
    print(i, now - last, 'ms')
    last = now
    for j in range(len(texts)):
        x = check(texts[i], texts[j])
        y = kernel[ids[i], ids[j]]

        

def clean_text(text):
    ne_char = chr(9000)
    blank_char = chr(10000)

    replace_pairs = [['$NE$', ne_char], ['ţ', 'ț'], ['ş', 'ș'], ['Ţ', 'Ț'], ['Ş', 'Ș'], ['Ṣ', 'Ș'], ['ṣ', 'ș'],
                    ['…', '. . .'], ['„', '"'], ['“', '"'], ['”', '"'], ['ã', 'ă'], ['–', '-'], ['—', '-'], ['’', "'"],
                    ['‘', "'"], ['`', "'"], ['″', '"'], ['ˮ', '"'], ['′', "'"], ['‚', ','], ['−', '-'], ['‑', '-'],
                    ['‐', '-']]
    
    remove_chars = ['\t', '»', ']', '[', 'и', 'р', 'с', 'н', '\xad', 'т', '>', 'в', '•', '|', 'л', '<', '😀',
                    'к', 'д', '̦', 'у', '̆', 'я', '·', 'м', 'п', 'ь', 'é', '\u200b', 'з', 'г', 'ы', '̂', 'й',
                    '\\', '^', 'Ă', 'С', 'ȋ', 'б', 'ж', '̧', '˝', 'ш', 'ч', 'В', 'М', 'à', 'К', '▪', '°',
                    'х', 'П', 'Ç', '€', 'á', 'Ø', 'ц', 'ü', 'Т', 'А', 'И', '«', '►', 'Р', 'Б', 'Н', '️',
                    'Á', 'Е', 'ю', '♦', '●', 'О', 'ö', 'Д', 'ё', '²', 'ȃ', '\uf0fc', 'ǎ', '±', 'ê', '×',
                    'Л', 'Ш', 'ф', '❤', '‒', 'Ф', 'Э', 'ğ', 'э', '⦁', 'ó', 'З', 'У', 'Ö', '§', 'щ', 'è',
                    'Я', 'É', 'İ', '\uf0d8', 'č', 'Ч', 'Х', 'า', 'í', 'Ȋ', 'α', '\u202a', '\u200e', 'Й',
                    'น', '\u200d', '♂', '⃣', '✅', 'і', 'μ', 'Г', 'Š', 'ı', 'ľ', 'ร', 'ห', 'ก', '🐟', '💦',
                    'Ł', '\u202c', '№', 'å', '\ufeff', 'Ж', 'ṭ', '³', '~', 'Å', '🤗', 'š', '¼', 'Ц', '⃰',
                    'Ы', 'ι', '\u2063', 'ᵁ', 'ᴸ', 'ᵀ', 'ᴿ', 'ᴬ', '®', 'Ć', 'ć', 'ต', 'บ', 'ธ', 'ุ', 'ล', 'ี',
                    'ด', 'ิ', '้', 'อ', 'ย', 'ง', 'ม', '🐠', '🐬', '🐳', 'ß', 'ź', '✔', 'ӑ', '˙', '✊', 'ç', 'ﬁ',
                    'Ü', '⚾', '❄', '�', 'Č', 'ä', 'Ю', '©', 'ń', 'ė', '⚽', 'Ō', 'Ñ', 'ž', 'ï', 'е', 'о', 'а']
    
    keep_only = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2',
                 '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                 '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', 'Î', 'â', 'î', 'ă', 'Ș', 'ș', 'Ț', 'ț', '⌨']


    for pair in replace_pairs:
        text = text.replace(pair[0], pair[1])

    pattern = re.compile(f"[^{''.join(keep_only)}]")

    text = re.sub(pattern, blank_char, text)

    return text


chars = {}
for text in texts:
    text = clean_text(text)

    for ch in list(text):
        if ch not in chars:
            chars[ch] = 0
        chars[ch] += 1

# chars.sort()
chars = [[k, v] for k, v in chars.items()]
chars.sort(key=lambda x: -x[1])

keep_chars = list(map(lambda x: x[0], chars))
# for ch_index in range(ord('a'), ord('z') + 1):
#     ch = chr(ch_index)
#     keep_chars.remove(ch)
#     keep_chars.remove(ch.upper())

keep_chars.sort()
print(keep_chars)

import pdb; pdb.set_trace()


