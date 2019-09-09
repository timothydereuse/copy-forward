import get_prediction as gp
import parse_csvs as pc
import argparse
import os

parser = argparse.ArgumentParser(description='Entry for the 2019 MIREX Patterns for prediction event.')
parser.add_argument('-i', dest='input_path', help='Path to folder containing .csvs with primes to process.')
parser.add_argument('-o', dest='output_path', help='Path to existing folder where predicted continuations will be saved.')
parser.add_argument('-l', dest='limit', default=-1, help='Process no more than this number of primes from the input folder. (Default: process all files in the input folder)')
parser.add_argument('-q', dest='quiet', action='store_true', help='Suppress console output.')
args = parser.parse_args()

out_path = args.output_path
verbose = not args.quiet

if verbose:
    print('Parsing primes...')
ids, data = pc.parse_prime_csvs(args.input_path, limit=int(args.limit))


for i, name in enumerate(ids):

    if not(i % 50) and verbose:
        print(f'processing file {i} of {len(ids)}...')

    prime = data[name][:, :2]
    prediction = gp.get_prediction(prime, bounds=None)
    out_fname = os.path.join(out_path, f'{name}.csv')
    pc.write_to_csv(prediction, out_fname)
