import os
import json
import csv

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
                arr.append(row)
            data[i]['cont'] = arr
        path = f'{PPDD}/prime_csv/{i}.csv'
        with open(path) as file:
            arr = []
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                arr.append(row)
            data[i]['prime'] = arr
    return data
