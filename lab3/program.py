import scipy.stats as stat
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import seaborn as sb

def mean(sample):
    return np.mean(sample)

def median(sample):
    return np.median(sample)

def z_R(sample):
    return (sample[0] + sample[-1]) / 2

def z_p(sample, p):
    pn = p * sample.size
    if (pn == int(pn)):
        return sample[int(pn)]
    return sample[int(pn) + 1]

def z_Q(sample):
    return (z_p(sample, 1/4) + z_p(sample, 3/4)) / 2

def cut_mean(sample):
    n = sample.size
    r = int(n / 4)
    sum = 0
    for i in range(r + 1, n - r + 1):
        sum += sample[i]
    return sum / (n - 2*r)

def variance(sample):
    return np.std(sample)**2


def lab_3():
    plt.style.use("seaborn")
    size = [20, 100]
    distr = [stat.norm, stat.cauchy, stat.laplace, stat.poisson, stat.uniform]
    name = ["normal", "koshi", "laplace", "poisson", "uniform"]
    colors = [["springgreen", "green"], ['tomato', 'red'], ['goldenrod', 'orangered'],
              ['pink', 'magenta'], ['turquoise', 'blue']]
    params = [[1, 0], [1, 0], [1 / np.sqrt(2), 0], [10], [2*np.sqrt(3), -np.sqrt(3)]]
    left_bound_def = -5
    right_bound_def = -left_bound_def
    for i in range(len(distr)):
        fig, axis = plt.subplots(len(size))
        fig.subplots_adjust(hspace=0.6)
        for j in range(len(size)):
            if len(params[i]) == 2:
                res = distr[i].rvs(scale=params[i][0], loc=params[i][1], size=size[j])
            else:
                res = distr[i].rvs(params[i][0], size=size[j])
            res.sort()
            sb.boxplot(data=res, color=colors[i][j], orient='h', ax=axis[j])
            axis[j].set_xlabel(str(size[j]) + " numbers")
        fig.savefig(name[i] + "Box.pdf")
    plt.show()


def outliers():
    distr = [stat.norm, stat.cauchy, stat.laplace, stat.poisson, stat.uniform]
    name = ["normal", "koshi", "laplace", "poisson", "uniform"]
    params = [[1, 0], [1, 0], [1 / np.sqrt(2), 0], [10], [2 * np.sqrt(3), -np.sqrt(3)]]
    size = [20, 100]
    for i in range(len(distr)):
        for j in range(len(size)):
            out_total = 0
            for k in range(1000):
                if len(params[i]) == 2:
                    res = distr[i].rvs(scale=params[i][0], loc=params[i][1], size=size[j])
                else:
                    res = distr[i].rvs(params[i][0], size=size[j])
                res.sort()
                z_min = z_p(res, 1/4)
                z_max = z_p(res, 3/4)
                out_1 = np.where(res <= (z_min - 1.5 * (z_max - z_min)))
                out_2 = np.where(res >= (z_max + 1.5 * (z_max - z_min)))
                out_total += len(out_2[0]) + len(out_1[0])
            print(name[i] + ' ' + str(size[j]) + ': ' + "%g" % (out_total / (1000 * size[j])))



outliers()

