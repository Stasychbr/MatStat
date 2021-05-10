import numpy as np
import matplotlib.pyplot as plot
from scipy.ndimage import median_filter
import os
from sklearn.decomposition import PCA
import pandas as pd

def read_data(filename):
    with open(filename) as file:
        lines = file.readlines()
        if lines[3].find('X/Y') == -1:
            print('file ' + filename + ' has strange format!')
            return None
        line = lines[3].removeprefix('X/Y	').replace(',', '.')
        x = [float(n) for n in line.split('\t')]
        y = []
        matrix = np.zeros((len(lines) - 4, len(x)))
        for i in range(4, len(lines)):
            nums = lines[i].replace(',', '.').split('\t')
            y.append(float(nums[0]))
            for j in range(1, len(nums)):
                matrix[i - 4, j - 1] = float(nums[j])
        x, y = y, x # swap
        matrix = matrix.T
        return x, y, matrix

def get_data(files_list):
    z_samples = []
    for filename in files_list:
        x, y, z_matrix = read_data(filename)
        '''
        grids = np.meshgrid(x, y)
        fig = plot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grids[0], grids[1], z_matrix, cmap='Spectral')
        fig = plot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grids[0], grids[1], median_filter(z_matrix, footprint=np.ones((10, 6)), mode='constant'),
                        cmap='Spectral')
        plot.show()
        '''
        z_samples.append(np.ravel(median_filter(z_matrix, footprint=np.ones((10, 6)), mode='constant')))
        #z_samples.append(np.ravel(z_matrix))
    return x, y, z_samples


def get_pca_res(samples, res_shape):
    pca = PCA()
    pca_data = pca.fit_transform(np.array(z_samples).T.tolist())
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['c' + str(x) for x in range(1, len(per_var) + 1)]
    plot.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    pca_df = pd.DataFrame(pca_data, columns=labels)
    xy_grids = np.meshgrid(x, y)
    comp1_grid = np.reshape(pca_df.c1.to_numpy(), res_shape)
    comp2_grid = np.reshape(pca_df.c2.to_numpy(), res_shape)
    return comp1_grid, comp2_grid


files_list = filter(lambda str: str.endswith('.txt'), os.listdir('./'))
x, y, z_samples = get_data(files_list)
xy_grids = np.meshgrid(x, y)
comp1_grid, comp2_grid = get_pca_res(z_samples, xy_grids[0].shape)
fig = plot.figure()
ax = fig.add_subplot()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(xy_grids[0], xy_grids[1], comp1_grid, cmap='Spectral')
cs = ax.contourf(xy_grids[0], xy_grids[1], comp1_grid, cmap='Spectral', levels=50)
ax.set_xlabel('Emission, nm.')
ax.set_ylabel('Excitation, nm.')
plot.colorbar(cs)
fig = plot.figure()
ax = fig.add_subplot()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(xy_grids[0], xy_grids[1], comp2_grid, cmap='Spectral')
cs = ax.contourf(xy_grids[0], xy_grids[1], comp2_grid, cmap='Spectral', levels=50)
ax.set_xlabel('Emission, nm.')
ax.set_ylabel('Excitation, nm.')
plot.colorbar(cs)
plot.show()
