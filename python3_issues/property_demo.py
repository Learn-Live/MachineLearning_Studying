# -*- coding:utf-8 -*-
r"""
    Purppose:
        # TODO
"""
__doc__ == 'property_demo'


class PropertyDemo(object):
    def __init__(self, temperature):
        """

        :param temperature:
        """
        self._temp = temperature

    def to_fahrenheit(self):
        """

        :return:
        """
        return (self.temp * 1.8) + 32

    @property
    def temperature(self):
        print('getter')
        return self._temp

    @temperature.setter
    def temperature(self, value):
        print('setter')
        if value < -273:
            raise ValueError('Temperature below -273 is not possible')
        self._temp = value


def main():
    """

    :return:
    """
    pd_ist = PropertyDemo(10.)
    print(pd_ist.__dict__)
    print(pd_ist.__dir__())
    pd_ist._temp = -20
    print(pd_ist._temp)
    print(pd_ist.temperature)

    print(__file__)
    print(__name__)
    print(__doc__)
    print(__import__)
    print(__package__)
    # print(__loader__.path)


if __name__ == '__main__':
    main()
