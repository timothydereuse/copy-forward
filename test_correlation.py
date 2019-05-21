import numpy as np
import parse_PPDD
import matplotlib.pyplot as plt
from importlib import reload
reload(parse_PPDD)

# remember:
# 0: onset time in beats
# 1: MIDI note
# 2: morphetic pitch estimation
# 3: duration in beats
# 4: channel
ids, data = parse_PPDD.parse_PPDD()

i = ids[811]
roll = data[i]['prime']
cont = data[i]['cont']
# parse_PPDD.plot_roll(roll, cont, mult=1)

def find_best_translation(roll, cont, window_size=0):

    cont_length = max(cont[:, 0]) - min(cont[:, 0])
    if not window_size:
        window_size = cont_length * 5

    # get bounds of window to compare against
    right_limit = max(roll[:, 0])
    left_limit = min(roll[:, 0])



    max_translate = right_limit - window_size - left_limit
    hop_translate = 2
    min_translate = 2
    translate_amts = list(range(min_translate, max_translate, hop_translate))

    window_base = roll[right_limit - window_size < roll[:, 0]]
    scores = []
    accuracies = []

    for t in translate_amts:
        roll_trans = np.copy(roll) # ELIMINATE THIS LATER!
        roll_trans[:, 0] = roll_trans[:, 0] + t
        window_trans = roll_trans[right_limit - window_size < roll_trans[:, 0]]
        window_trans = window_trans[window_trans[:, 0] < right_limit]
        score = rolls_match(window_base, window_trans)
        scores.append(score)

        # find prediction and compare against test
        prediction = roll_trans[right_limit < roll_trans[:, 0]]
        prediction = prediction[prediction[:, 0] < right_limit + cont_length]
        accuracy = rolls_match(cont, prediction)
        accuracies.append(accuracy)

    return scores, accuracies

def rolls_match(ra, rb):
    # just consider onset and pitch right now

    score = 0

    for f in [1]:
        ra_set = {tuple(x // f) for x in ra[:, (0, 1)]}
        rb_set = {tuple(x // f) for x in rb[:, (0, 1)]}
        size = len(ra_set.intersection(rb_set)) / len(ra_set.union(rb_set))
        score += size

    return size

scores, acc = find_best_translation(roll, cont)
plt.clf()
plt.plot(scores)
plt.plot(acc)
plt.show()
