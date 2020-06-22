import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


value = np.array([1, 2, 1, 1, 2, 3, 2, 2, 1])

print(value)
print(softmax(value))
print(value / np.sum(value))

plt.subplot(2,1,1)
plt.plot(value, 'r.-')
plt.subplot(2,1,2)
plt.plot(softmax(value), 'g.-')
plt.plot(value / np.sum(value), 'b.-')
plt.show()
