# -*- coding:utf-8 -*-
r"""
    sort by words length
"""
from collections import OrderedDict


def load_data(input_file):
    words_dict = OrderedDict()
    with open(input_file, 'r') as in_hdl:

        for line in in_hdl:
            if line.strip().startswith('#'):
                continue

            word = (line.strip('\n')).lower()
            if word not in words_dict.keys():
                words_dict[word] = len(word)
            else:
                print(f'\'{word}\' already in the words_dict.')

    return words_dict


def sort_words_length(words_dict):
    res = [(k, words_dict[k]) for k in sorted(words_dict, key=words_dict.get, reverse=True)]
    line = ''

    for idx, (k, v) in enumerate(res):
        line += k + ':' + str(v) + '\n'

        if idx % 10 == 0:
            line += '\n'
    # print(res)

    return line


def save_data(data_str, output_file='sorted_words.txt'):
    with open(output_file, 'w') as out_hdl:
        out_hdl.write(data_str)

    return output_file


def main():
    input_file = 'wiki-100k.txt'
    words_dict = load_data(input_file)
    print(f"{len(words_dict.keys())}")
    data_str = sort_words_length(words_dict)
    save_data(data_str, output_file='sorted_words.txt')


if __name__ == '__main__':
    main()
