import os
import csv
import numpy as np
import json

default_multiplier = 12


def parse_prime_csvs(in_path, multiplier=default_multiplier, limit=np.inf):

    if limit < 0:
        limit = np.inf

    i = 0
    ids = []
    data = {}
    for root, directories, filenames in os.walk(in_path):
        for filename in filenames:
            ext = filename.split('.')[-1]
            if not ext == 'csv':
                continue
            path = os.path.join(root, filename)
            id = filename.split('.')[0]
            ids.append(id)
            with open(path) as f:
                reader = csv.reader(f)
                entry = []
                for row in reader:
                    time = int(float(row[0]) * multiplier)
                    note = int(row[1])
                    entry.append([time, note])
                data[id] = np.array(entry)
                i += 1
            if i >= limit:
                break
    return ids, data


def parse_PPDD(PPDD='PPDD', limit=1000, mult=default_multiplier):

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


def write_to_csv(prediction, path, multiplier=default_multiplier, round_to=5):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        for pt in prediction:
            time = np.round(pt[0] / float(multiplier), round_to)
            note = int(pt[1])
            w.writerow([time, note])
