import numpy as np
import parse_PPDD
import matplotlib.pyplot as plt
from importlib import reload
import evaluate_prediction as ep
reload(parse_PPDD)
reload(ep)

multiplier = 8
cont_length_default = multiplier * 10


def find_best_translation(prime, cont, bounds=None, window_size=None):

    # get time-bounds of the prime
    if not bounds:
        right_limit = max(prime[:, 0])
        left_limit = min(prime[:, 0])
    else:
        right_limit = bounds[1]
        left_limit = bounds[0]

    cont_length = cont_length_default
    if not window_size:
        window_size = min(cont_length // 2, (right_limit - left_limit - cont_length) // 2)

    max_translate_steps = int(right_limit - left_limit - cont_length - window_size)
    hop_translate = 1
    min_translate = 0
    translate_amts = np.arange(min_translate, max_translate_steps, hop_translate)

    window_base = prime[right_limit - window_size < prime[:, 0]]
    scores = {'rec': [], 'prec': [], 'F1': []}
    accuracies = {'rec': [], 'prec': [], 'F1': []}

    best_prediction = np.array([])
    best_score = 0
    for t in translate_amts:

        window_slide_l = left_limit + t
        window_slide_r = left_limit + t + window_size
        inds_slide = np.logical_and(window_slide_l <= prime[:, 0], prime[:, 0] < window_slide_r)
        window_slide = np.copy(prime[inds_slide, :])

        # distance from start of sliding window to start of fixed window
        window_start_diff = right_limit - window_size - window_slide_l
        window_slide[:, 0] = window_slide[:, 0] + window_start_diff

        autocorrelation = rolls_match(window_base, window_slide)
        # autocorrelation = ep.evaluate_tec(window_base, window_slide)
        for key in autocorrelation:
            scores[key].append(autocorrelation[key])

        predict_l = window_slide_r
        predict_r = window_slide_r + cont_length
        predict_inds = np.logical_and(predict_l <= prime[:, 0], prime[:, 0] < predict_r)
        prediction = np.copy(prime[predict_inds, :])

        # translate to start of true continuation
        prediction[:, 0] = prediction[:, 0] + right_limit - predict_r

        if len(cont) == 0 and len(prediction) == 0:
            accuracy = {'rec': 1, 'prec': 1, 'F1': 1}
        elif len(cont) == 0 or len(prediction) == 0:
            accuracy = {'rec': 0, 'prec': 0, 'F1': 0}
        else:
            try:
                accuracy = ep.evaluate_tec(cont, prediction)
                # print(f'translation {t} stats: {accuracy}')
            except ZeroDivisionError:
                accuracy = {'rec': 0, 'prec': 0, 'F1': 0}
                # print(f'zero division on translation {t}')
        for key in accuracy:
            accuracies[key].append(accuracy[key])

        if accuracy['F1'] > best_score:
            best_score = accuracy['F1']
            best_prediction = prediction

    return scores, accuracies, best_prediction


def rolls_match(orig, pred):
    # just consider onset and pitch right now

    intersect_total = 0

    or_set = {tuple(x) for x in orig[:, (0, 1)]}
    pr_set = {tuple(x) for x in pred[:, (0, 1)]}
    intersect_total += len(or_set.intersection(pr_set))
    or_total_size = len(or_set)
    pr_total_size = len(pr_set)

    if or_total_size == 0 and pr_total_size == 0:
        return {'rec': 1, 'prec': 1, 'F1': 1}

    recall = intersect_total / or_total_size if or_total_size > 0 else 0
    precision = intersect_total / pr_total_size if pr_total_size > 0 else 0
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

    for idx in range(200,250):
        # get as triples of (onset time, pitch, channel)
        i = ids[idx]
        prime = data[i]['prime'][:, [0, 1, 4]]
        cont = data[i]['cont'][:, [0, 1, 4]]

        bounds = (min(prime[:, 0]), max(prime[:, 0]))

        channel_nums = set(prime[:, 2])
        best_predictions = []

        for channel in channel_nums:
            channel_prime = prime[prime[:, 2] == channel, :2]
            channel_cont = cont[cont[:, 2] == channel, :2]
            scores, acc, best_prediction = find_best_translation(channel_prime, channel_cont, bounds=bounds)
            best_predictions.append(best_prediction)
            best_trans_predicted = np.argmax(scores['F1'])
            best_trans_actual = np.argmax(acc['F1'])
            best_possible = np.round(max(scores['F1']), 3)
            print(f'channel {channel}, predicted translation = {best_trans_predicted}, best translation = {best_trans_actual} '
                  f'best possible score = {best_possible}')

        mixed_prediction = np.concatenate([x for x in best_predictions if x.size > 0])
        res = ep.evaluate_tec(cont[:2], mixed_prediction)
        print(res)

    print('plotting...')
    parse_PPDD.plot_roll(data[i])
    # plt.figure(2)
    # plt.plot(scores['F1'])
    # plt.plot(acc['F1'])
    # plt.legend(['autocorrelation', 'accuracy'])
    # plt.show()
