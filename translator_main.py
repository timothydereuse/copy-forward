import get_prediction as gp
import parse_PPDD as p
import argparse

parser = argparse.ArgumentParser(description='Entry for the 2019 MIREX Patterns for prediction event.')
parser.add_argument(dest='input_path', help='Path to folder containing .csvs with primes to process.')
parser.add_argument(dest='output_path', help='Path to folder where predicted continuations will be saved.')
parser.add_argument('-l', dest='limit', default=-1, help='Process no more than this number of primes from the input folder.')

args = parser.parse_args()