class Animal:

    def __init__(self):
        self.value = 5
        self.value_2 = 6

    def __call__(self, *args, **kwargs):
        print(args)

    def __repr__(self):
        print('__repr__ :{self.value}')
        a = self.value

        return str(a)


def func(val_str=''):
    print(f'{val_str} func')


def main():
    anim_inst = Animal()
    print(callable(anim_inst))
    res_val = anim_inst('callable')
    print(callable(func))
    print(type(func))
    print(func.__call__)
    let = func  # python's functions are first-class objects. you can assign them to variable, store them in data structures,
    # pass them as arguments to other functions, and even return them as values from other functions
    print(callable(let))
    let('hi')

    print(anim_inst)


if __name__ == '__main__':
    main()
