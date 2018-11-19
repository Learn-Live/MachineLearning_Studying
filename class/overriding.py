r"""
    https://www.programcreek.com/2009/02/overriding-and-overloading-in-java-with-examples/

    overriding:
        same method name and same parameter
        e.g. add() - > add()  : for example (from the Latin exempli gratia )

    overloading:
        same method name and different parameter
        e.g. add() -> add(a, b)
"""

class Animals:

    def eat(self):
        print('animals')


class Dog(Animals):

    def __init__(self):
        pass

    def eat(self):
        i = 2
        print('Dog, overriding', i)

    def eat(self, food=''):
        print('overloading', food)


def add_values(a, b):
    return a + b


def add_values(a, b, c):
    return a + b + c


add_values(1, 2)
# What looks like overloading methods, it is actually that Python keeps only the latest definition of a method you declare to it.
# This code doesn’t make a call to the version of add() that takes in two arguments to add.
# So we find it safe to say Python doesn’t support method overloading.

if __name__ == '__main__':
    dog = Dog()
    dog.eat()
