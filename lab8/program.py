import numpy as np
import scipy.stats as stats
from scipy.ndimage import median_filter
import seaborn as sb
import matplotlib.pyplot as plt

def read_cluster(k):
    file = open('wave_ampl.txt', 'r')
    line = file.readline()[1:-1]
    arr = [float(x) for x in line.split(',')]
    sample = arr[1024*(k-1):1024*k]
    file.close()
    return sample

def save_signal_fig(signal, k, postfix = ''):
    x = [i for i in range(1024)]
    plt.plot(x, signal)
    plt.title('Сигнал ' + str(k))
    plt.xlabel('Момент времени')
    plt.ylabel('Амплитуда')
    plt.savefig('signal' + str(k) + postfix + '.pdf')


def smooth_signal(signal):
    return median_filter(signal, size=5)


def get_hist_fig(signal):
    return sb.histplot(signal, stat="count", bins=10)


def get_intervals(axis):
    bars = axis.containers[0].patches
    bars.sort(key=lambda rect: -rect.get_height())
    noise = (bars[0].get_x(), bars[0].get_x() + bars[0].get_width())
    signal = (bars[1].get_x(), bars[1].get_x() + bars[1].get_width())
    trans = [(bars[i].get_x(), bars[i].get_x() + bars[i].get_width()) for i in range(2, len(bars))]
    return noise, signal, trans


def distribute_points(signal):
    axis = get_hist_fig(signal)
    noise_int, signal_int, trans_int = get_intervals(axis)
    all_noise_points = []
    noise_points = [[], []]
    signal_points = [[], []]
    all_trans_points = []
    trans_points = [[], []]
    for i in range(len(signal)):
        if noise_int[0] <= signal[i] <= noise_int[1]:
            if len(noise_points[0]) > 0 and i - noise_points[0][-1] > 1:
                all_noise_points.append(noise_points)
                noise_points = [[], []]
            noise_points[0].append(i)
            noise_points[1].append(signal[i])
        elif signal_int[0] <= signal[i] <= signal_int[1]:
            signal_points[0].append(i)
            signal_points[1].append(signal[i])
        else:
            if len(trans_points[0]) > 0 and i - trans_points[0][-1] > 1:
                all_trans_points.append(trans_points)
                trans_points = [[], []]
            trans_points[0].append(i)
            trans_points[1].append(signal[i])
    all_noise_points.append(noise_points)
    all_trans_points.append(trans_points)
    return all_noise_points, signal_points, all_trans_points


def fisher(noise_dom, signal, trans_dom):
    groups = [noise_dom[0][1], trans_dom[0][1], signal[1], trans_dom[1][1], noise_dom[1][1]]
    splits = [7, 4, 4, 4, 7]
    k = len(splits)
    fisher = np.zeros(k)
    for k in range(len(splits)):
        inc = len(groups[k]) // splits[k] + 1
        cur_subsample = [groups[k][i:i+inc] for i in range(0, len(groups[k]), inc)]
        inter_group = 0
        full_mean = np.mean([np.mean(cur_subsample[j]) for j in range(splits[k])])
        for j in range(splits[k]):
            inter_group += (np.mean(cur_subsample[j]) - full_mean) ** 2
        inter_group *= splits[k] / (splits[k] - 1)
        inta_group = 0
        for i in range(splits[k]):
            for l in range(len(cur_subsample[i])):
                inta_group += (cur_subsample[i][l] - np.mean(cur_subsample[i])) ** 2
        inta_group /= splits[k] * (splits[k] - 1)
        fisher[k] = inter_group / inta_group
    return fisher


plt.style.use("seaborn")
signal = read_cluster(228)
signal = smooth_signal(signal)
noise, signal, trans = distribute_points(signal)
print(fisher(noise, signal, trans))




