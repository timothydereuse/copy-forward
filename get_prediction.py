import numpy as np
import parse_PPDD
import matplotlib.pyplot as plt
from importlib import reload
import os
import csv
import evaluate_prediction as ep
from collections import Counter
reload(parse_PPDD)
reload(ep)

multiplier = 12
cont_length_default = multiplier * 10


def get_best_translation(prime_window, fixed_window):
    translation_vectors = []
    generated_vec = np.array([(float(s[0]), int(s[1])) for s in fixed_window])
    prime_list = [(float(s[0]), int(s[1])) for s in prime_window]
    for i in prime_list:
        vectors = generated_vec - i
        translation_vectors += [tuple(v) for v in vectors]
    grouped_vectors = dict(Counter(translation_vectors))
    best_vector = max(grouped_vectors, key=lambda x: grouped_vectors[x])
    return best_vector, grouped_vectors[best_vector]


def extract_by_time_range(inp, left, size):
    inds = np.logical_and(left <= inp[:, 0], inp[:, 0] < left + size)
    return np.copy(inp[inds, :])


def get_prediction(prime, bounds=None, window_size=None):

    # if there are no notes in the prime - why bother?
    if len(prime) == 0:
        return (0, 0), ((0, 0), (0, 0)), ([], [])

    if not bounds:
        left_limit = min(prime[:, 0])
        right_limit = max(prime[:, 0])
    else:
        right_limit = bounds[1]
        left_limit = bounds[0]

    cont_length = cont_length_default

    # if the prime spans less time than the desired continuation then repeat the prime backwards
    if right_limit - left_limit < cont_length:
        factor = int(np.ceil(cont_length / (right_limit - left_limit)))
        old_prime = np.copy(prime)
        for i in range(1, factor + 1):
            translate_prime = np.copy(old_prime)
            translate_prime[:, 0] -= (right_limit - left_limit) * i
            prime = np.concatenate([translate_prime, prime])

    if not window_size:
        window_size = min(cont_length // 2, (right_limit - left_limit) // 2)

    fixed_window = extract_by_time_range(prime, right_limit - window_size, window_size)
    prime_window = extract_by_time_range(prime, 0, right_limit - window_size)
    # fixed_window = prime[right_limit - window_size <= prime[:, 0]]
    # prime_window = prime[prime[:, 0] < right_limit - window_size]

    # if the fixed window has no notes in it - there's nothing to go on. assume that the continuation is also empty
    if len(fixed_window) == 0 or len(prime_window) == 0:
        best_trans_vector = (0, 0)
        predicted_cont = []
    else:
        best_trans_vector, best_amt = get_best_translation(prime_window, fixed_window)
        translated_prime = prime + best_trans_vector
        predicted_cont = extract_by_time_range(translated_prime, right_limit, cont_length)

    return predicted_cont


def write_to_csv(prediction, path, multiplier, round_to=5):
    with open(path, 'w') as f:
        w = csv.writer(f)
        for pt in prediction:
            time = np.round(pt[0] / float(multiplier), round_to)
            note = int(pt[1])
            w.writerow([time, note])


if __name__ == '__main__':
    ids, data = parse_PPDD.parse_prime_csvs('./PPDD/cont_true_csv', multiplier=multiplier, limit=100)
    out_path = './out_csv/'

    print('translating...')
    for idx in range(len(ids)):
        i = ids[idx]
        prime = data[i][:, :2]
        prediction = get_prediction(prime, bounds=None, window_size=cont_length_default // 2)
        out_fname = os.path.join(out_path, f'{i}.csv')
        write_to_csv(prediction, out_fname, multiplier=multiplier)