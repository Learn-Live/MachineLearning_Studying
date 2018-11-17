import platform

print(platform.platform())


def _import_dotted_name(name):
    components = name.split('.')
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj


if __name__ == '__main__':
    res = _import_dotted_name('b.a')
    print(res)
