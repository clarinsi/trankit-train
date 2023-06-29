import argparse
import time
import conllu


def read_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input", default='data/ssj/sl_ssj-ud-dev.conllu', type=str, help="conllu input")
    parser.add_argument("--output", default='data/ssj/dev.txt', type=str, help="text output")
    return parser.parse_args()


def main():
    args = read_args()
    data_file = open(args.input, "r", encoding="utf-8")
    text_file = ''
    for tokenlist in conllu.parse_incr(data_file):
        if text_file and 'newdoc_id' in tokenlist.metadata:
            text_file += '\n'
        if text_file and 'newpar_id' in tokenlist.metadata:
            text_file += '\n'
        text_file += tokenlist.metadata['text'] + ' '
    data_file.close()
    with open(args.output, 'w', encoding='utf-8') as wf:
        wf.write(text_file)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total:")
    print("--- %s seconds ---" % (time.time() - start_time))
