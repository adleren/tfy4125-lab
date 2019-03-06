from modules.iptrack import iptrack
from modules.trvalues import trvalues

import numpy as np
from matplotlib import pyplot as plt

# Max utslag
coeffsMAX = iptrack('data/max/track1')

coeffsTauto = iptrack('data/Tautochron/Tautochron.txt')

coeffsskraa = iptrack('data/Skråplan/Skråplan.txt')





def f(x, coeffs):
	return trvalues(coeffs, x)[0]

x = np.linspace(0, 1.3, 101)

plt.plot(x, f(x,coeffsskraa))
plt.plot(x,f(x,coeffsMAX))
plt.plot(x,f(x,coeffsTauto))
plt.show()
