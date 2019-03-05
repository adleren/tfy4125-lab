from modules.iptrack import iptrack
from modules.trvalues import trvalues

import numpy as np
from matplotlib import pyplot as plt

# Max utslag
coeffs = iptrack('data/max/track1')



def f(x):
	return trvalues(coeffs, x)[4]

x = np.linspace(0, 1.3, 101)

plt.plot(x, f(x))
plt.show()
	