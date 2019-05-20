import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np

def parse_PPDD():
    PPDD = 'PPDD-Jul2018_sym_poly_medium'

    i = 0
    data = {}
    for root, directories, filenames in os.walk(f'{PPDD}/descriptor'):
        for filename in filenames:
            path = os.path.join(root, filename)
            with open(path) as json_file:
                entry = json.load(json_file)
                data[entry['id']] = entry

    ids = list(data.keys())

    for i in ids:
        path = f'{PPDD}/cont_true_csv/{i}.csv'
        with open(path) as file:
            arr = []
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                arr.append(np.array(row, dtype='float'))
            data[i]['cont'] = np.array(arr)
        path = f'{PPDD}/prime_csv/{i}.csv'
        with open(path) as file:
            arr = []
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                arr.append(np.array(row, dtype='float'))
            data[i]['prime'] = np.array(arr)
    return ids, data


def plot_roll(roll, roll2=[]):
    x = roll[:, 0]
    y = roll[:, 1]
    c = roll[:, 4].astype('int')

    last = max(roll[:, 0] + roll[:, 3])

    colors = np.array(['k', 'b', 'g', 'r', 'c', 'm', 'y'])

    if not roll2 is None:
        x = np.concatenate((x, roll2[:, 0]))
        y = np.concatenate((y, roll2[:, 1]))
        c = np.concatenate((c, roll2[:, 4].astype('int')))

    c = c % len(colors)

    plt.clf()
    plt.axvline(last)
    plt.scatter(x,y,c=colors[c])
    plt.show()
