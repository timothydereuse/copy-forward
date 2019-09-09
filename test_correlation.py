import numpy as np
import parse_csvs as pc
# import matplotlib.pyplot as plt
from importlib import reload
import evaluate_prediction as ep
from collections import Counter
reload(pc)
reload(ep)

multiplier = 12
cont_length_default = multiplier * 10
window_size_default = multiplier * 8


def get_best_translation(prime_window, fixed_window):
    # based on the evaluate_tec method from
    # github.com/BeritJanssen/PatternsForPrediction.
    translation_vectors = []
    generated_vec = np.array(fixed_window)
    for i in prime_window:
        vectors = generated_vec - i
        translation_vectors += [tuple(v) for v in vectors]
    grouped_vectors = dict(Counter(translation_vectors))
    best_vector = max(grouped_vectors, key=lambda x: grouped_vectors[x])
    return best_vector, grouped_vectors[best_vector]


def extract_by_time_range(inp, left, size):
    inds = np.logical_and(left <= inp[:, 0], inp[:, 0] < left + size)
    return np.copy(inp[inds, :])


def get_all_translations(prime, cont, bounds, window_size=None):

    # if there are no notes in the prime - why bother?
    if len(prime) == 0:
        return (0, 0), ((0, 0), (0, 0)), ([], [])

    right_limit = bounds[1]
    left_limit = bounds[0]

    cont_length = cont_length_default

    # if the prime spans less time than the desired continuation then just repeat the prime backwards
    if right_limit - left_limit < cont_length:
        factor = int(np.ceil(cont_length / (right_limit - left_limit)))
        old_prime = np.copy(prime)
        for i in range(1, factor + 1):
            translate_prime = np.copy(old_prime)
            translate_prime[:, 0] -= (right_limit - left_limit) * i
            prime = np.concatenate([translate_prime, prime])

    if not window_size:
        window_size = min(cont_length // 2, (right_limit - left_limit) // 2)

    fixed_window = prime[right_limit - window_size <= prime[:, 0]]
    prime_window = prime[prime[:, 0] < right_limit - window_size]

    # if the fixed window has no notes in it - there's nothing to go on. assume that the continuation is also empty
    if len(fixed_window) == 0 or len(prime_window) == 0:
        best_trans_vector = (0, 0)
        predicted_cont = []
    else:
        best_trans_vector, best_amt = get_best_translation(prime_window, fixed_window)
        translated_prime = prime + best_trans_vector
        predicted_cont = extract_by_time_range(translated_prime, right_limit, cont_length)

    # if the continuation has no notes in it BUT the fixed window does: well, we're gonna be 100% wrong no matter what
    if len(cont) == 0:
        predicted_score = 0
        ideal_score = 0
        ideal_trans_vector = 0
        ideal_cont = []
    else:
        ideal_trans_vector, ideal_amt = get_best_translation(prime, cont)
        ideal_translated_prime = prime + ideal_trans_vector
        ideal_cont = extract_by_time_range(ideal_translated_prime, right_limit, cont_length)

        predicted_score = rolls_match(cont, predicted_cont)['F1']
        ideal_score = rolls_match(cont, ideal_cont)['F1']

    return (predicted_score, ideal_score), (best_trans_vector, ideal_trans_vector), (predicted_cont, ideal_cont)


def rolls_match(orig, pred):
    # just consider onset and pitch right now

    if len(orig) == 0 and len(pred) == 0:
        return {'rec': 1, 'prec': 1, 'F1': 1}
    elif len(orig) == 0 or len(pred) == 0:
        return {'rec': 0, 'prec': 0, 'F1': 0}

    intersect_total = 0

    or_set = {tuple(x) for x in orig[:, (0, 1)]}
    pr_set = {tuple(x) for x in pred[:, (0, 1)]}
    intersect_total += len(or_set.intersection(pr_set))
    or_total_size = len(or_set)
    pr_total_size = len(pr_set)

    recall = intersect_total / or_total_size
    precision = intersect_total / pr_total_size
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * recall * precision) / (
            recall + precision
        )

    return {'rec': recall, 'prec': precision, 'F1': f1}


def plot_rolls(rolls):
    rolls[:, :, 1] *= 2
    rolls = np.sum(rolls, axis=2)
    plt.imshow(rolls.T)
    plt.show()


if __name__ == '__main__':
    # remember:
    # 0: onset time in beats
    # 1: MIDI note
    # 2: morphetic pitch estimation
    # 3: duration in beats
    # 4: channel
    print('parsing PPDD...')
    ids, data = pc.parse_PPDD(PPDD='./PPDD', limit=10000, mult=multiplier)

    pred_accs = []
    ideal_accs = []
    m_pred_accs = []
    m_ideal_accs = []

    print('translating...')
    indices_choose = np.random.choice(10000, 1000)
    for n, idx in enumerate(indices_choose):

        if not n % 50:
            print(f'processing entry {n} of {len(indices_choose)}...')

        # get as triples of (onset time, pitch, channel)
        i = ids[idx]
        prime = data[i]['prime'][:, [0, 1, 4]]
        cont = data[i]['cont'][:, [0, 1, 4]]

        bounds = (min(prime[:, 0]), max(prime[:, 0]))

        # prime[:, 2] = 0
        # cont[:, 2] = 0
        channel_nums = list(set(prime[:, 2]))

        # channel_lengths = [len(prime[prime[:,2] == x, :2]) for x in channel_nums]
        # merge_channels = [channel_nums[i] for i, x in enumerate(channel_lengths) if x < np.mean(channel_lengths)]
        #
        # if len(merge_channels) > 1:
        #     for c in merge_channels:
        #         prime[prime[:, 2] == c, 2] = -1
        #     channel_nums = list(set(prime[:, 2]))

        # best_predictions = []
        # ideal_predictions = []
        #
        # pred_scores = []
        # for channel in channel_nums:
        #     channel_prime = prime[prime[:, 2] == channel, :2]
        #     channel_cont = cont[cont[:, 2] == channel, :2]
        #
        #     scores, vectors, continuations = \
        #         get_all_translations(channel_prime, channel_cont, bounds=bounds, window_size=window_size_default // 2)
        #     best_predictions.extend(continuations[0])
        #     ideal_predictions.extend(continuations[1])
        #     # print(f'c {channel}, pred. trans = {vectors[0]}, ideal trans = {vectors[1]} '
        #     #       f'pred score = {scores[0]:.3f}')
        #     pred_scores.append(scores[0] * len(channel_prime))

        # pred_avg_score = np.mean(pred_scores) / len(prime)
        m_scores, m_vectors, m_continuations = \
            get_all_translations(prime[:, :2], cont[:, :2], bounds=bounds, window_size=window_size_default // 2)

        # best_predictions = sorted(best_predictions, key=lambda x: x[0])
        # if len(best_predictions) > 0:
        #     best_predictions = np.unique([tuple(x) for x in best_predictions], axis=0)
        #
        # ideal_predictions = sorted(ideal_predictions, key=lambda x: x[0])
        # if len(ideal_predictions) > 0:
        #     ideal_predictions = np.unique([tuple(x) for x in ideal_predictions], axis=0)

        if len(m_continuations[0]) > 0:
            m_best_predictions = np.unique([tuple(x) for x in m_continuations[0]], axis=0)

        if len(m_continuations[1]) > 0:
            m_ideal_predictions = np.unique([tuple(x) for x in m_continuations[1]], axis=0)

        mixed_true = cont[:, :2]
        mixed_true = np.unique([tuple(x) for x in mixed_true], axis=0)

        if len(m_best_predictions) == 1:
            m_best_predictions = np.concatenate([m_best_predictions, [[0, 0]]])
        if len(m_ideal_predictions) == 1:
            m_ideal_predictions = np.concatenate([m_ideal_predictions, [[0, 0]]])

        try:
            # res_pred = ep.evaluate_tec(mixed_true, best_predictions)['F1']
            # res_ideal = ep.evaluate_tec(mixed_true, ideal_predictions)['F1']
            m_res_pred = ep.evaluate_tec(mixed_true, m_best_predictions)
            m_res_ideal = ep.evaluate_tec(mixed_true, m_ideal_predictions)

            # pred_accs.append(res_pred)
            # ideal_accs.append(res_ideal)
            m_pred_accs.append(m_res_pred)
            m_ideal_accs.append(m_res_ideal)
        except ValueError:
            print('empty prediction - continuing')
            continue

        # better = ((res_pred < m_res_pred) == (pred_avg_score < m_scores[0])) or (res_pred == m_res_pred)

        # print(
        #     f'pred: {res_pred:.3f}. m_pred: {m_res_pred:.3f}, diff: {res_pred - m_res_pred:.3f} scorediff: {pred_avg_score - m_scores[0]:.3f} better: {better}'
        # )

        # print(
        #     f'm_pred: {m_res_pred}'
        # )

    mean_res_pred = {}
    mean_res_ideal = {}
    for k in m_pred_accs[0].keys():
        mean_res_pred[k] = np.mean([x[k] for x in m_pred_accs])
        mean_res_ideal[k] = np.mean([x[k] for x in m_ideal_accs])

    print(mean_res_pred)
    print(mean_res_ideal)

    # print(f'mean: {np.mean(m_pred_accs):4f}, std_err: {np.std(m_pred_accs) / np.sqrt(len(m_pred_accs)):4f}')
    # print(f'ideal_mean: {np.mean(m_ideal_accs):4f}, ideal_std_err: {np.std(m_ideal_accs) / np.sqrt(len(m_ideal_accs)):4f}')

    plt.clf()
    print('plotting...')
    pc.plot_roll(data[i])
    plt.figure(2)
    plt.scatter([x[0] for x in mixed_true], [x[1] for x in mixed_true], facecolors='none', edgecolors='k', s=80)
    plt.scatter([x[0] for x in m_ideal_predictions], [x[1] for x in m_ideal_predictions], marker='o')
    plt.scatter([x[0] for x in m_best_predictions], [x[1] for x in m_best_predictions], marker='s', s=30)
    # plt.plot(acc['F1'])
    plt.legend(['truth', 'ideal', 'predicted'])
    plt.show()
