import numpy as np
import parse_PPDD
import matplotlib.pyplot as plt
from importlib import reload
import evaluate_prediction as ep
reload(parse_PPDD)
reload(ep)

multiplier = 12
cont_length_default = multiplier * 10


def find_best_translation(prime, cont, window_size=None):

    cont_length = cont_length_default
    if not window_size:
        window_size = cont_length // 2

    # get time-bounds of the prime
    right_limit = max(prime[:, 0])
    left_limit = min(prime[:, 0])

    max_translate = int(right_limit - left_limit - cont_length - window_size)
    hop_translate = 1
    min_translate = 0
    translate_amts = np.arange(min_translate, max_translate, hop_translate)

    window_base = prime[right_limit - window_size < prime[:, 0]]
    scores = {'rec': [], 'prec': [], 'F1': []}
    accuracies = {'rec': [], 'prec': [], 'F1': []}

    print(right_limit, left_limit, len(translate_amts))
    for t in translate_amts:

        window_slide_l = left_limit + t
        window_slide_r = left_limit + t + window_size
        inds_slide = np.logical_and(window_slide_l <= prime[:, 0], prime[:, 0] < window_slide_r)
        window_slide = np.copy(prime[inds_slide, :])

        #distance from start of sliding window to start of fixed window
        window_start_diff = right_limit - window_size - window_slide_l
        window_slide[:, 0] = window_slide[:, 0] + window_start_diff

        autocorrelation = rolls_match(window_base, window_slide)
        for key in autocorrelation:
            scores[key].append(autocorrelation[key])

        predict_l = window_slide_r
        predict_r = window_slide_r + cont_length
        predict_inds = np.logical_and(predict_l <= prime[:, 0], prime[:, 0] < predict_r)
        prediction = np.copy(prime[predict_inds, :])

        # translate to start of true continuation
        prediction[:, 0] = prediction[:, 0] + right_limit - predict_r

        try:
            accuracy = ep.evaluate_tec(cont, prediction)
            # print(f'translation {t} stats: {accuracy}')
        except ZeroDivisionError:
            accuracy = {'rec': 0, 'prec': 0, 'F1': 0}
            # print(f'zero division on translation {t}')
        for key in accuracy:
            accuracies[key].append(accuracy[key])

    return scores, accuracies


def rolls_match(orig, pred):
    # just consider onset and pitch right now

    score = 0

    or_total_size = 0
    pr_total_size = 0
    intersect_total = 0

    for f in [1]:
        or_set = {tuple(x // f) for x in orig[:, (0, 1)]}
        pr_set = {tuple(x // f) for x in pred[:, (0, 1)]}
        intersect_total += len(or_set.intersection(pr_set))
        or_total_size += len(or_set)
        pr_total_size += len(pr_set)

    recall = intersect_total / or_total_size
    precision = intersect_total / pr_total_size
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * recall * precision) / (
            recall + precision
        )

    return {'rec': recall, 'prec': precision, 'F1': f1}


if __name__ == '__main__':
    # remember:
    # 0: onset time in beats
    # 1: MIDI note
    # 2: morphetic pitch estimation
    # 3: duration in beats
    # 4: channel
    print('parsing PPDD...')
    ids, data = parse_PPDD.parse_PPDD(limit=1000, mult=multiplier)

    print('translating...')
    i = ids[199]
    prime = data[i]['prime'][:, :2]
    cont = data[i]['cont'][:, :2]

    scores, acc = find_best_translation(prime, cont)

    print('plotting...')
    parse_PPDD.plot_roll(data[i])
    plt.figure(2)
    plt.plot(scores['F1'])
    plt.plot(acc['F1'])
    plt.legend(['autocorrelation', 'accuracy'])
    plt.show()
