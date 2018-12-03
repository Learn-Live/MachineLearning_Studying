# https://stackoverflow.com/questions/35758323/hook-python-module-function

# import whatever
#
#
# def prefix_function(function, prefunction):
#     @functools.wraps(function)
#     def run(*args, **kwargs):
#         prefunction(*args, **kwargs)
#         return function(*args, **kwargs)
#
#     return run
#
#
# def this_is_a_function(parameter):
#     pass  # Your own code here that will be run before
#
#
# whatever.this_is_a_function = prefix_function(
#     whatever.this_is_a_function, this_is_a_function)

# def decorator(argument):
#     def real_decorator(function):
#         def wrapper(*args, **kwargs):
#             funny_stuff()
#             something_with_argument(argument)
#             result = function(*args, **kwargs)
#             more_funny_stuff()
#             return result
#         return wrapper
#     return real_decorator


import time


# def calculate_time(name='calculate_time'):
#     def calculate_time_1(func,name):
#         def wrapper(*args, **kwargs):
#             st_time= time.time()
#             res=func(*args, **kwargs)
#             ed_time = time.time()
#             print(f"{name},: {ed_time-st_time}")
#             return res
#         return wrapper
#     return calculate_time_1
#
# def calculate_time_decorator(func,message='mess'):
#
#     def wrapper(*args, **kwargs):
#         st_time = time.time()
#         res=func(*args, **kwargs)
#         ed_time = time.time()
#         print(f"{message},: {ed_time-st_time}")
#         return res
#
#     return wrapper


def calculate_time_decorator(func, message='mess'):
    def new_func(*args, **kwargs):
        st_time = time.time()
        res = func(*args, **kwargs)
        ed_time = time.time()
        print(f"{message},: {ed_time-st_time}")

        return res

    return new_func


@calculate_time_decorator
def process_data(x, y):
    print(f"x:{x}")

    return -1


if __name__ == '__main__':
    # process_data(10,1)
    foo = calculate_time_decorator(process_data, 'process_data')
    foo(10, 1)
