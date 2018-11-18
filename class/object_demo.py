r"""
    # https://amir.rachum.com/blog/2016/10/03/understanding-python-class-instantiation/

    Classes, functions, methods and instances are all objects and whenever you put parentheses after their name,
    you invoke their __call__ method.


    So Foo(1, y=2) is equivalent to Foo.__call__(1, y=2). That __call__ is the one defined by Foo’s class. What is Foo’s class?
"""


# If we ignore error checking for a minute, then for regular class instantiation this is roughly equivalent to:

def __call__(obj_type, *args, **kwargs):
    obj = obj_type.__new__(*args, **kwargs)
    if obj is not None and issubclass(obj, obj_type):
        obj.__init__(*args, **kwargs)
    return obj

# # __new__ allocates memory for the object, constructs it as an “empty” object and then __init__ is called to initialize it.
#
# In conclusion:
#
#     Foo(*args, **kwargs) is equivalent to Foo.__call__(*args, **kwargs).
#     Since Foo is an instance of type, Foo.__call__(*args, **kwargs) calls type.__call__(Foo, *args, **kwargs).
#     type.__call__(Foo, *args, **kwargs) calls type.__new__(Foo, *args, **kwargs) which returns obj.
#     obj is then initialized by calling obj.__init__(*args, **kwargs).
#     obj is returned.


