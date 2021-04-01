import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import seaborn as sb

def Pirson(sel):
    return stats.pearsonr(np.transpose(np.transpose(sel)[0]), np.transpose(sel)[1])[0]


def Spirman(sel):
    return stats.spearmanr(sel)[0]


def r_q(sel):
    n = [0, 0, 0, 0]
    for s in sel:
        if s[0] > 0 and s[1] > 0:
            n[0] += 1
        elif s[0] > 0 and s[1] < 0:
            n[1] += 1
        elif s[0] < 0 and s[1] < 0:
            n[2] += 1
        elif s[0] < 0 and s[1] > 0:
            n[3] += 1
    return (n[0] + n[2] - n[1] - n[3]) / len(sel)

def double_dim():

    sizes = [20, 60, 100]
    rhos = [0, 0.5, 0.9]

    for size in sizes:
        print('###########')
        print('n = ', size)
        # fig, axes = plt.subplots(nrows=1, ncols=3)
        # fig.subplots_adjust(wspace=0.5, top=0.93, bottom=0.175)
        # fig.set_figwidth(9)
        # fig.set_figheight(3)
        file = open('tables/table' + str(size) + '.txt', 'w')
        file.write('\\begin{tabular}{|c|c|c|c|}\n')
        file.write('\\hline\n')
        for k in range(len(rhos)):
            rho = rhos[k]
            print('---------')
            print('rho = ', rho)
            pirs = []
            rq = []
            spir = []
            for i in range(1000):
                sel = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).rvs(size=size)
                pirs.append(Pirson(sel))
                spir.append(Spirman(sel))
                rq.append(r_q(sel))
            file.write('\\multicolumn{4}{|c|}{$\\rho=' + str(rho) + '$}\\\\\n')
            file.write('\\hline\n')
            file.write('&$r$&$r_S$&$r_Q$\\\\\n')
            file.write('\\hline\n')
            print('E(z)')
            p_r = round(np.mean(pirs), 4)
            print('Pirson = ', p_r)
            p_r_s = round(np.mean(spir), 4)
            print('Spirman = ', p_r_s)
            p_r_q = round(np.mean(rq), 4)
            print('r_q = ', p_r_q)
            file.write('E($z$)&' + str(p_r) + '&' + str(p_r_s) + '&' + str(p_r_q) + '\\\\\n')
            file.write('\\hline\n')
            print('E(z^2)')
            p_r = round(np.mean([a ** 2 for a in pirs]), 4)
            print('Pirson = ', p_r)
            p_r_s = round(np.mean([a ** 2 for a in spir]), 4)
            print('Spirman = ', p_r_s)
            p_r_q = round(np.mean([a ** 2 for a in rq]), 4)
            print('r_q = ', p_r_q)
            file.write('E($z^2$)&' + str(p_r) + '&' + str(p_r_s) + '&' + str(p_r_q) + '\\\\\n')
            file.write('\\hline\n')
            print('D(z)')
            p_r = round(np.var(pirs), 4)
            print('Pirson = ', p_r)
            p_r_s = round(np.var(spir), 4)
            print('Spirman = ', p_r_s)
            p_r_q = round(np.var(rq), 4)
            print('r_q = ', p_r_q)
            file.write('D($z$)&' + str(p_r) + '&' + str(p_r_s) + '&' + str(p_r_q) + '\\\\\n')
            file.write('\\hline\n')

            # angle = np.pi / 4
            # a = np.sqrt(np.cos(angle) ** 2 + rho + np.sin(angle) ** 2) * 3
            # b = np.sqrt(np.sin(angle) ** 2 - rho + np.cos(angle) ** 2) * 3
            # t = np.linspace(0, 2 * np.pi, 1000)
            # ellipse = np.array([a * np.cos(t), b * np.sin(t)])
            # A = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # for i in range(ellipse.shape[1]):
            #     ellipse[:,i] = A.dot(ellipse[:, i])
            # axes[k].plot(ellipse[0,:], ellipse[1,:], 'g--')
            # axes[k].scatter(sel[:, 0], sel[:, 1])
            # axes[k].set_xlabel('$\\rho$ = ' + str(rho))
        file.write('\\end{tabular}')
        file.close()
        #fig.savefig('plots/plot' + str(size) + '.pdf')
    #plt.show()

def mix():
    sizes = [20, 60, 100]
    file = open('tables/tableMix.txt', 'w')
    file.write('\\begin{tabular}{|c|c|c|c|}\n')
    file.write('\\hline\n')
    for size in sizes:
        pirs = []
        rq = []
        spir = []
        for i in range(1000):
            selection_1 = stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size)
            selection_2 = stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)
            sel = 0.9 * selection_1 + 0.1 * selection_2
            pirs.append(Pirson(sel))
            spir.append(Spirman(sel))
            rq.append(r_q(sel))
        file.write('\\multicolumn{4}{|c|}{$n=' + str(size) + '$}\\\\\n')
        file.write('\\hline\n')
        file.write('&$r$&$r_S$&$r_Q$\\\\\n')
        file.write('\\hline\n')
        print('E(z)')
        p_r = round(np.mean(pirs), 4)
        print('Pirson = ', p_r)
        p_r_s = round(np.mean(spir), 4)
        print('Spirman = ', p_r_s)
        p_r_q = round(np.mean(rq), 4)
        print('r_q = ', p_r_q)
        file.write('E($z$)&' + str(p_r) + '&' + str(p_r_s) + '&' + str(p_r_q) + '\\\\\n')
        file.write('\\hline\n')
        print('E(z^2)')
        p_r = round(np.mean([a ** 2 for a in pirs]), 4)
        print('Pirson = ', p_r)
        p_r_s = round(np.mean([a ** 2 for a in spir]), 4)
        print('Spirman = ', p_r_s)
        p_r_q = round(np.mean([a ** 2 for a in rq]), 4)
        print('r_q = ', p_r_q)
        file.write('E($z^2$)&' + str(p_r) + '&' + str(p_r_s) + '&' + str(p_r_q) + '\\\\\n')
        file.write('\\hline\n')
        print('D(z)')
        p_r = round(np.var(pirs), 4)
        print('Pirson = ', p_r)
        p_r_s = round(np.var(spir), 4)
        print('Spirman = ', p_r_s)
        p_r_q = round(np.var(rq), 4)
        print('r_q = ', p_r_q)
        file.write('D($z$)&' + str(p_r) + '&' + str(p_r_s) + '&' + str(p_r_q) + '\\\\\n')
        file.write('\\hline\n')
    file.write('\\end{tabular}')
    file.close()

mix()