arr = [23, 34, 56, 21, 21, 56, 78, 23, 34]

s = 0
for i, x in enumerate(arr):
    s = s ^ x
    print(i, s)
print(s)
