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


def lab_1():
    plt.style.use("seaborn")
    size = [10, 50, 1000]
    distr = [stat.norm, stat.cauchy, stat.laplace, stat.poisson, stat.uniform]
    name = ["normal", "koshi", "laplace", "poisson", "uniform"]
    colors = [["springgreen", "darkgreen"], ['tomato', 'red'], ['goldenrod', 'orangered'],
              ['pink', 'magenta'], ['turquoise', 'navy']]
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
            left_bound = min(min(res), left_bound_def) if name[i] != 'poisson' else 0
            right_bound = max(max(res), right_bound_def)
            sb.histplot(res, stat="density", ax=axis[j], color=colors[i][0])
            x = np.linspace(left_bound, right_bound, 1000 if name[i] != 'poisson' else right_bound + 1)
            if name[i] == 'laplace' or name[i] == 'uniform':
                y = distr[i](scale=params[i][0], loc=params[i][1]).pdf(x)
            elif name[i] == 'poisson':
                y = distr[i](params[i][0]).pmf(x)
            else:
                y = distr[i].pdf(x)
            if name[i] == 'poisson':
                axis[j].plot(x, y, 'o', color=colors[i][1])
            else:
                axis[j].plot(x, y, color=colors[i][1])
            axis[j].set_xlabel(str(size[j]) + " numbers")
        fig.savefig(name[i] + ".pdf")
    plt.show()

def lab_2():
    number_of_experiments = 1000
    units = [10, 100, 1000]
    E = []
    D = []
    for u_num in units:
        samples_means = []
        samples_medians = []
        samples_z_Rs = []
        samples_z_Qs = []
        samples_z_trs = []
        for i in range(number_of_experiments):
            sample = stat.uniform.rvs(loc=-3**0.5, scale=2*3**0.5,size=u_num)
            samples_means.append(mean(sample))
            samples_medians.append(median(sample))
            sample.sort()
            samples_z_Rs.append(z_R(sample))
            samples_z_Qs.append(z_Q(sample))
            samples_z_trs.append(cut_mean(sample))
        val_lists = [samples_means, samples_medians, samples_z_Rs, samples_z_Qs, samples_z_trs]
        E_s = [round(mean(val_list), 6) for val_list in val_lists]
        D_s = [round(variance(val_list), 6) for val_list in val_lists]
        #print(E_s)
        #print(D_s)
        #print('\n')
        E.append(E_s)
        D.append(D_s)
    return E, D

# E, D = lab_2()
# for i in range(3):
#     n = 10 ** (i+1)
#     print("n = " + str(n) + ': ')
#     string = ''
#     for j in range(len(E[i])):
#         string += '&' + str(E[i][j])
#     print('$E(z)$' + string + '\\\\')
#     string = ''
#     for j in range(len(D[i])):
#         string += '&' + str(D[i][j])
#     print('$D(z)$' + string + '\\\\')

string = 'a'
while len(string) > 0:
    e = float(input())
    d = float(input())
    d = np.sqrt(d)
    print(e + d, e - d)
    string = input()
