import os
import numpy as np
import pypianoroll as pypr
from matplotlib import pyplot as plt

def load_rand_rolls(num, from_first):
    i = 0
    paths = []
    for root, directories, filenames in os.walk('lpd_5'):
        for filename in filenames:
            paths.append(os.path.join(root, filename))

        i += 1
        if i > from_first:
            break

    chosen_paths = np.random.choice(paths, num, replace=False)
    rolls = [pypr.load(p) for p in chosen_paths]
    return rolls

if __name__ == '__main__':
    rolls = load_rand_rolls(10,500)
    fig, ax = rolls[5].plot()
    plt.show()
