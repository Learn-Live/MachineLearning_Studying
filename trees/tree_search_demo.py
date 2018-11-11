def binary_search(input_l='lst', low=0, high=10 - 1, search_num=''):
    """

    :param input_l: the lst must be a sorted lst
    :param low:
    :param hihg:
    :param search_num:
    :return:
    """
    if low > high:
        print('cannot find %s in this lst' % search_num)
        return False

    mid = (low + high) // 2
    if input_l[mid] == search_num:
        print('%s can be found in this lst' % search_num)
        return True
    elif input_l[mid] < search_num:
        binary_search(input_l, low=low, hihg=mid - 1, search_num=search_num)
    else:
        binary_search(input_l, low=mid + 1, high=high - 1, search_num=search_num)


def binary_search_with_recursion(input_l='lst', search_num=''):
    """

    :param input_l: the lst must be a sorted lst
    :param low:
    :param hihg:
    :param search_num:
    :return:
    """
    low = 0
    high = len(input_l) - 1

    if low > high:
        print('cannot find %s in this lst' % search_num)
        return False

    mid = (low + high) // 2
    if input_l[mid] == search_num:
        print('%s can be found in this lst' % search_num)
        return True
    elif input_l[mid] > search_num:
        binary_search_with_recursion(input_l[low:mid], search_num=search_num)
    else:
        binary_search_with_recursion(input_l[mid + 1:high], search_num=search_num)


def binary_search_with_iteration(input_l='lst', search_num=''):
    """

    :param input_l: the lst must be a sorted lst
    :param search_num:
    :return:
    """

    low = 0
    high = len(input_l)

    while low < high:
        # mid = (low + high) // 2  # overflow
        mid = low + (high - low) // 2
        if input_l[mid] == search_num:
            print('%s can be found in this lst' % search_num)
            return True
        elif input_l[mid] < search_num:
            low = mid + 1
        else:
            high = mid - 1
    else:
        print('cannot find %f in this lst' % search_num)
    return False


if __name__ == '__main__':
    input_l = [1, 2, 4, 5, 7, 7, 9, 10]
    search_num = 3
    binary_search_with_iteration(input_l, search_num=7)
    binary_search_with_recursion(input_l, search_num=7)
    binary_search(input_l=input_l, search_num=3)
