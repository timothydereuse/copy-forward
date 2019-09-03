import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np


def parse_PPDD(limit=1000, mult=12):
    PPDD = 'PPDD'

    i = 0
    data = {}
    for root, directories, filenames in os.walk(f'{PPDD}/descriptor'):
        for filename in filenames:
            ext = filename.split('.')[-1]
            if not ext == 'json':
                continue
            path = os.path.join(root, filename)
            with open(path) as json_file:
                entry = json.load(json_file)
                data[entry['id']] = entry
                i += 1
            if i >= limit:
                break

    ids = list(data.keys())

    for i in ids:
        path = f'{PPDD}/cont_true_csv/{i}.csv'
        with open(path) as file:
            cont = []
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                cont.append(np.array(row, dtype='float'))
            cont = np.array(cont)
        path = f'{PPDD}/prime_csv/{i}.csv'
        with open(path) as file:
            prime = []
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                prime.append(np.array(row, dtype='float'))
            prime = np.array(prime)

        prime[:, 0] = np.round(prime[:, 0] * mult)
        prime[:, 3] = np.round(prime[:, 3] * mult)
        cont[:, 0] = np.round(cont[:, 0] * mult)
        cont[:, 3] = np.round(cont[:, 3] * mult)

        data[i]['prime'] = prime.astype('int16')
        data[i]['cont'] = cont.astype('int16')

    # multiply and round to get integer values for time
    return ids, data


def plot_roll(inp, roll2=None, mult=12):

    if type(inp) == dict:
        roll = inp['prime']
        roll2 = inp['cont']
    else:
        roll = inp

    x = roll[:, 0]
    y = roll[:, 1]
    c = roll[:, 4].astype('int')

    last = max(roll[:, 0])

    colors = np.array(['k', 'b', 'g', 'r', 'c', 'm', 'y'])

    if roll2 is not None:
        x = np.concatenate((x, roll2[:, 0]))
        y = np.concatenate((y, roll2[:, 1]))
        c = np.concatenate((c, roll2[:, 4].astype('int')))

    c = c % len(colors)
    x = x / mult
    last = last / mult

    plt.figure(5)
    plt.clf()
    plt.axvline(last)
    plt.scatter(x, y, c=colors[c])
