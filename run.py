from modules.iptrack import iptrack
from modules.trvalues import trvalues
from math import pi
import numpy as np
from matplotlib import pyplot as plt

# Max utslag
coeffsMAX = iptrack('data/max/max.txt')

coeffsTauto = iptrack('data/tautochrone/tautochrone.txt')

coeffsskraa = iptrack('data/slope/slope.txt')

def accel(alpha):
    g = 9.81
    c = 2 / 3  #constant of moment of inertia for a hollow sphere

    return (g * np.sin(alpha)) / (1 + c)


def vel(vn, a, h):
    return vn + h * a


def posX(xn, h, vn, alpha):
    return xn + h * vn * np.cos(alpha)


def integrate(coeffs):
    vn = 0
    xn = 0.1
    h = 1e-3
    m = 0.0027
    tn = 0
    xlist = [xn]
    vlist = [vn]
    tlist = [tn]
    friclist = [0]
    normlist = [0]

    while xn < 1.30:
        alpha = trvalues(coeffs, xn)[3]
        R = trvalues(coeffs, xn)[4]
        a = accel(alpha)
        vn = vel(vn, a, h)
        xn = posX(xn, h, vn, alpha)
        tn += h
        vlist.append(vn)
        xlist.append(xn)
        tlist.append(tn)
        friclist.append((2 / 3) * m * 9.81 * np.sin(alpha))
        normlist.append(m * 9.81 * np.cos(alpha) + m * vn**2 / R)

    return tlist, xlist, vlist, normlist, friclist


def plotPosX(filename):
    x = np.loadtxt(filename, skiprows=2)[:, 1]
    t = np.loadtxt(filename, skiprows=2)[:, 0]
    return [t, x]


def plotVel(filename):
    xdata = np.loadtxt(filename, skiprows=2)[:, 1]
    tdata = np.loadtxt(filename, skiprows=2)[:, 0]
    ydata = np.loadtxt(filename, skiprows=2)[:, 2]
    vlist = [0]

    for i in range(1, len(xdata)):
        v = np.sqrt((xdata[i] - xdata[i - 1])**2 +
                    (ydata[i] - ydata[i - 1])**2) / (tdata[i] - tdata[i - 1])
        vlist.append(v)
    return [tdata, vlist]


def plotAll(filename, coeffs):
    tlist, xlist, vlist, normlist, friclist = integrate(coeffs)
    velo = plotVel(filename)
    pos = plotPosX(filename)
    fig = plt.figure()
    print("TID", tlist[-1])
    print("X", xlist[-1])

    plt.subplot(2, 2, 1)
    plt.subplots_adjust(hspace=0.8, wspace=0.6)
    plt.plot(tlist, xlist, "--")
    plt.plot(pos[0], pos[1])
    plt.title("Posisjon")
    plt.xlabel("Tid [s]")
    plt.ylabel("X-posisjon [m]")
    plt.legend([r"Numerisk", r"Eksperimentell"], prop={'size': 6})
    plt.subplot(2, 2, 2)
    plt.plot(tlist, vlist, "--")
    plt.plot(velo[0], velo[1])
    plt.title("Fart")
    plt.ylabel("Hastighet i x-retning [m/s]")
    plt.xlabel("Tid [s]")
    plt.legend([r"Numerisk", r"Experimentell"], prop={'size': 6})
    plt.subplot(2, 2, 3)

    plt.plot(tlist, normlist)
    plt.title("Normalkraft")
    plt.xlabel("Tid [s]")
    plt.ylabel("Kraft [N]")
    plt.subplot(2, 2, 4)
    plt.plot(tlist, friclist)
    plt.title("Friksjon")
    plt.xlabel("Tid [s]")
    plt.ylabel("Kraft [N]")
    plt.show()


plotAll('data/Skråplan/Skråplan.txt', coeffsskraa)
plotAll('data/Tautochron/Tautochron.txt', coeffsTauto)
plotAll('data/max/testing2', coeffsMAX)
x = np.linspace(0.03, 1.3, 1000)

#tlist, vlist = plotPosX('data/Tautochron/Tautochron.txt')
#tlist2, vlist2 = plotPosX('data/max/testing2')
#plt.plot(tlist, vlist)
#plt.plot(tlist2, vlist2)
#plt.plot(tlist2, [1.3] * len(tlist2))
#plt.legend([r"Tauto", r"Max"])
#plt.show()


def g(x):
    return 360 * trvalues(coeffsTauto, x)[3] / pi


f, ax = plt.subplots(1)

#ax.plot(x, g(x))
#ax.set_ylim(ymin=0)
#plt.show(f)