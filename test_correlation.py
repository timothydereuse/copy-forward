import numpy as np
import parse_PPDD
import matplotlib.pyplot as plt
from importlib import reload
import evaluate_prediction as ep
reload(parse_PPDD)
reload(ep)


def find_best_translation(roll, cont, window_size=None):

    cont_length = max(cont[:, 0]) - min(cont[:, 0])
    if not window_size:
        window_size = cont_length * 5

    # get bounds of window to compare against
    right_limit = max(roll[:, 0])
    left_limit = min(roll[:, 0])

    max_translate = int(right_limit - window_size - left_limit)
    hop_translate = 1
    min_translate = 1
    translate_amts = np.arange(min_translate, max_translate, hop_translate)

    window_base = roll[right_limit - window_size < roll[:, 0]]
    scores = {'rec': [], 'prec': [], 'F1': []}
    accuracies = {'rec': [], 'prec': [], 'F1': []}

    for t in translate_amts:
        roll_trans = np.copy(roll)
        roll_trans[:, 0] = roll_trans[:, 0] + t
        window_trans = roll_trans[right_limit - window_size < roll_trans[:, 0]]
        window_trans = window_trans[window_trans[:, 0] < right_limit]
        score = rolls_match(window_base, window_trans)
        for key in score:
            scores[key].append(score[key])

        # find prediction and compare against test
        prediction = roll_trans[right_limit < roll_trans[:, 0]]
        prediction = prediction[prediction[:, 0] < right_limit + cont_length]
        try:
            accuracy = ep.evaluate_tec(cont, prediction)
        except ZeroDivisionError:
            accuracy = {'rec': 0, 'prec': 0, 'F1': 0}
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
    ids, data = parse_PPDD.parse_PPDD()

    print('translating...')
    i = ids[879]
    roll = data[i]['prime'][:, :2]
    cont = data[i]['cont'][:, :2]
    # parse_PPDD.plot_roll(roll, cont, mult=1)

    scores, acc = find_best_translation(roll, cont)

    print('plotting...')
    parse_PPDD.plot_roll(data[i])
    plt.figure(2)
    plt.plot(scores['F1'])
    plt.plot(acc['F1'])
    plt.legend(['autocorrelation', 'accuracy'])
    plt.show()
