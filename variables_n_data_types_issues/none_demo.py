a = None
print(a)
print(type(None))

# b= NoneType()

c = 1.0
print(type(c))
c_1 = float(1.0)
print(type(c_1))

"""
In python3 that types.NoneType is gone it directly type(None). So you can

either use type(None).
"""
NoneType = type(None)
b = NoneType()
print(type(b))
