# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from uncertainties import unumpy


def getLogNmValues(magnitudes, mCutoff=None):
    mags = unumpy.nominal_values(magnitudes)
    mMin, mMax = np.min(mags), np.max(mags)
    m = np.linspace(mMin, mMax, 100)
    N = np.array([np.sum(mags <= mi) for mi in m])

    # Remove points where there was only 1 count
    # Because then N - sqrt(N) = 0, and log(0) is undefined
    m = m[N > 1]
    N = N[N > 1]


    # Remove points above the specified magnitude threshold
    # to account for incompleteness
    if not (mCutoff is None):
        N = N[m < mCutoff]
        m = m[m < mCutoff]


    Nerr = np.sqrt(N)
    logN = np.log10(N)
    logNupper = np.log10(N + Nerr)
    logNlower = np.log10(N - Nerr)

    sigmaLow = logN - logNlower
    sigmaUpp = logNupper - logN


    return m, logN, sigmaLow, sigmaUpp



def plotLogNm(magnitudes):
    m, logN, sigmaLow, sigmaUpp = getLogNmValues(magnitudes)


    plt.figure(dpi=400)
    plt.errorbar(m, logN, (sigmaLow, sigmaUpp), fmt=".", label="data")
    plt.xlabel("Magnitude")
    plt.ylabel("log N(>m)")
    # plt.show()


# A straight line
def f(x, grad, c):
    return grad*x + c


def fitLogNm(magnitudes, mCutoff):
    m, logN, sigmaLow, sigmaUpp = getLogNmValues(magnitudes, mCutoff)



    # https://stackoverflow.com/questions/19116519/scipy-optimize-curvefit-asymmetric-error-in-fit

    def loss_function(params):
        error = (logN - f(m, *params))
        error_neg = (error < 0)
        error_squared = error**2 / (error_neg * sigmaLow + (1 - error_neg) * sigmaUpp)
        return error_squared.sum()

    x0 = [0.3, -3]
    x = fmin(loss_function, x0)
    print("log N(m) = %.4fm %.4f" % (x[0], x[1]))

    plotLogNm(magnitudes)
    # plt.plot(m, f(m, *x0), label="initial guess")
    plt.plot(m, f(m, *x), label="fit", zorder=100)
    plt.title("Number count plot")
    plt.legend()
    plt.show()


    return x
# %%


