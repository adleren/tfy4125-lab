import numpy as np

def euler(f, x_0, y_0, h, N):
	x = np.zeros(N+1)
	y = np.zeros(N+1)
	
	x[0] = x_0
	y[0] = y_0

	for n in range(N):
		x[n+1] = x[n] + h * f(x[n], y[n])
		y[n+1] = y[n] + h
	
	return x, y