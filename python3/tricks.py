# -*- coding: utf-8 -*-
"""

"""


def lst_enurmate(in_lst=''):
    for i, x in enumerate(in_lst):
        print(i, x)


def load_data(in_f=''):
    """

    :param in_f: input file
    :return:
    """
    try:
        with open(in_f, 'r') as read_h:  # read_handle
            for idx, line_str in enumerate(read_h):
                print(idx, line_str.strip('\n'))
        # return 0  # no matter what, finally always be execute
    except FileNotFoundError as e:  # like 'if '
        print('e:', e)
    else:
        print('try successful.')
    finally:
        print('print finally')

    print('body')


if __name__ == '__main__':
    # lst_enurmate(in_lst=[3,2,1,4])
    load_data(in_f='../data/attack_demo.csv')
