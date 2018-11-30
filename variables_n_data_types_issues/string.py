

x =10
printer = 'dell'

print('I just printed %s pages to the printer %s'%(x, printer))
# using the format method
print('I just printed {x} pages to the printer {printer}'.format(x=x,printer=printer))
print('I just printed {0} pages to the printer {1}'.format(x, printer))

# using f-strings.
print(f"I just printed {x} pages to the printer {printer}")
