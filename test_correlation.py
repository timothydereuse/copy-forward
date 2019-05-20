import numpy as np
import parse_PPDD
from importlib import reload
reload(parse_PPDD)

# remember:
# 0: onset time in beats
# 1: MIDI note
# 2: morphetic pitch estimation
# 3: duration in beats
# 4: channel
ids, data = parse_PPDD.parse_PPDD()

i = ids[100]
roll = ids[i]['prime']
cont = ids[i]['cont']
parse_PPDD.plot_roll(roll, cont)

def find_best_translation(roll, pred_length):

    left_limit = 0
    right_limit = max(roll[:, 0] + roll[:, 3])

    pass
