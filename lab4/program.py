import scipy.stats as stat
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import seaborn as sb



def distributions_4():
    plt.style.use("seaborn")
    size = [20, 60, 100]
    distr = [stat.norm, stat.cauchy, stat.laplace, stat.poisson, stat.uniform]
    name = ["normal", "koshi", "laplace", "poisson", "uniform"]
    colors = [["springgreen", "darkgreen"], ['tomato', 'red'], ['goldenrod', 'orangered'],
              ['pink', 'magenta'], ['turquoise', 'navy']]
    params = [[1, 0], [1, 0], [1 / np.sqrt(2), 0], [10], [2 * np.sqrt(3), -np.sqrt(3)]]
    left_bound_def = -5
    right_bound_def = -left_bound_def
    for i in range(len(distr)):
        fig, axis = plt.subplots(len(size))
        fig.subplots_adjust(hspace=0.6)
        for j in range(len(size)):
            if len(params[i]) == 2:
                sample = distr[i].rvs(scale=params[i][0], loc=params[i][1], size=size[j])
            else:
                sample = distr[i].rvs(params[i][0], size=size[j])
            left_bound = min(sample)
            right_bound = max(sample)
            sample.sort()
            distrFunc = [[], []]
            cumFreq = 0
            k = 0
            while k  < len(sample):
                curFreq = 1
                while k < len(sample) - 1 and sample[k] == sample[k+1]:
                    curFreq += 1
                    k += 1
                cumFreq += curFreq
                distrFunc[0].append(sample[k])
                distrFunc[1].append(cumFreq / len(sample))
                k += 1
            if name[i] != 'poisson':
                x = np.linspace(left_bound, right_bound, 1000)
            else:
                x = sample
            if name[i] == 'laplace' or name[i] == 'uniform':
                y = distr[i](scale=params[i][0], loc=params[i][1]).cdf(x)
            elif name[i] == 'poisson':
                y = distr[i](params[i][0]).cdf(x)
            else:
                y = distr[i].cdf(x)
            if name[i] != 'poisson':
                axis[j].plot(x, y, color=colors[i][0])
            else:
                axis[j].step(x, y, color=colors[i][0], where='post')
            axis[j].step(distrFunc[0], distrFunc[1], color=colors[i][1], where='post')
            axis[j].set_xlabel(str(size[j]) + " numbers")
        fig.savefig(name[i] + "Distr.pdf")
    #plt.show()


def kernel_4():
    plt.style.use("seaborn")
    size = [20, 60, 100]
    distr = [stat.norm, stat.cauchy, stat.laplace, stat.poisson, stat.uniform]
    name = ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]
    colors = [["springgreen", "darkgreen"], ['tomato', 'red'], ['goldenrod', 'orangered'],
              ['pink', 'magenta'], ['turquoise', 'navy']]
    params = [[1, 0], [1, 0], [1 / np.sqrt(2), 0], [10], [2 * np.sqrt(3), -np.sqrt(3)]]
    left_bound_basic = -4
    right_bound_basic = -left_bound_basic
    for i in range(len(distr)):
        if name[i] != 'Poisson':
            left_bound = left_bound_basic
            right_bound = right_bound_basic
        else:
            left_bound = 6
            right_bound = 14
        for j in range(len(size)):
            if len(params[i]) == 2:
                sample = distr[i].rvs(scale=params[i][0], loc=params[i][1], size=size[j])
            else:
                sample = distr[i].rvs(params[i][0], size=size[j])
            fig, axis = plt.subplots(3)
            fig.subplots_adjust(hspace=0.6)
            for m in range(3):
                kernel = 0
                h_n = 1.06 * np.std(sample) * np.power(size[j], -1/5)
                h_n *= 2 ** (m - 1)
                sample.sort()
                x = np.linspace(left_bound, right_bound, 1000)
                for k in range(size[j]):
                    kernel += stat.norm.pdf((x - sample[k]) / h_n)
                kernel = 1 / (size[j] * h_n) * kernel
                if name[i] == 'Laplace' or name[i] == 'Uniform':
                    y = distr[i](scale=params[i][0], loc=params[i][1]).pdf(x)
                elif name[i] == 'Poisson':
                    x_y = [i for i in range(left_bound, right_bound + 1)]
                    y = distr[i](params[i][0]).pmf(x_y)
                else:
                    y = distr[i].pdf(x)
                if name[i] == 'Poisson':
                    axis[m].plot(x_y, y, 'o', color=colors[i][0])
                    axis[m].plot(x, kernel, color=colors[i][1])
                else:
                    axis[m].plot(x, y, color=colors[i][0])
                    axis[m].plot(x, kernel, color=colors[i][1])
                if m == 0:
                    string = '$0.5 h_n$'
                elif m == 1:
                    string = '$h_n$'
                else:
                    string = '$2 h_n$'
                axis[m].set_xlabel(string)
            fig.suptitle(name[i] + ' ' + str(size[j]) + ' numbers')
            fig.savefig(name[i] + str(size[j]) + 'numbers' + "Kernel.pdf")
    #plt.show()

kernel_4()
