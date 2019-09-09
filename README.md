# CopyForward

A MIREX project for the 2019 Patterns for Prediction task:
`https://www.music-ir.org/mirex/wiki/2019:Patterns_for_Prediction`
by Timothy de Reuse.

Input Representations: symbolic, monophonic & symbolic, polyphonic

Subtasks: 1

Command line:

```
usage: copy_forward.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH] [-l LIMIT] [-q]

optional arguments:
  -h, --help      show this help message and exit
  -i INPUT_PATH   Path to folder containing .csvs with primes to process.
  -o OUTPUT_PATH  Path to existing folder where predicted continuations will be saved.
  -l LIMIT        Process no more than this number of primes from the input
                  folder (Default: process all files in the input folder).
  -q              Suppress console output.
```

Example usage:

```
python copy_forward.py -i ./PPDD-Sep2018_sym_poly_large/prime_csv -o ./copyforward_out -l 100

```

Each `.csv` file containing a prime will be processed and saved with the same name in the output folder; the file `input/foo.csv` will have its predicted continuation saved into `output/foo.csv`.

Will use only one thread. Memory footprint will be on the order of tens of megabytes (all input `.csv` files will be loaded into memory simultaneously). Takes about 15 minutes to process 10,000 primes on an ordinary laptop (1.6 GHz Intel i5).

Requires Python 3.7 and numpy >= 1.16.1.
