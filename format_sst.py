'''
A script that copy&pastes form to lemma if lemma is not given. Trankit returns error if that is the case.
'''

import argparse
import os
import time

from conllu import parse_incr


def read_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--conllu_input_fpath", type=str, default='data/ssj+sst/sl_ssj+sst-ud-train.conllu', help="input annotations file in CONLLU format for testing")
    parser.add_argument("--conllu_output_fpath", type=str, default='data/ssj+sst/sl_ssj+sst-ud-train-formatted.conllu', help="output annotations file in CONLLU format for testing")
    return parser.parse_args()


def main():
    args = read_args()
    if os.path.exists(args.conllu_output_fpath):
        os.remove(args.conllu_output_fpath)
    data_file = open(args.conllu_input_fpath, "r", encoding="utf-8")
    with open(args.conllu_output_fpath, 'a') as wf:
        for tokenlist in parse_incr(data_file):
            for token in tokenlist:
                if token['lemma'] == '_':
                    token['lemma'] = token['form']
            wf.write(tokenlist.serialize())

            # print(tokenlist)
    data_file.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total:")
    print("--- %s seconds ---" % (time.time() - start_time))