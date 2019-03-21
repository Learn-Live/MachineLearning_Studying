# import numpy as np
import matplotlib.pyplot as plt

x=[200,400,600,784,1500]
y=[0.848184818,0.841584158,0.843234323,0.84323,0.8399]
ax=plt.plot(x, y)
plt.ylim(0.5,1)
plt.ylabel('Accuracy')
plt.xlabel('The length of sessions')
plt.title('Random forest')
plt.show()


{0: 1086, 1: 341, 2: 445, 3: 692, 4: 591, 5: 287, 6: 546, 7: 829, 8: 701, 9: 220}