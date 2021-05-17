import numpy as np
import matplotlib.pyplot as plot
from scipy.ndimage import median_filter
import os
from sklearn.decomposition import PCA
import pandas as pd


# reading sample out of the file
def read_data(filename):
    with open(filename) as file:
        lines = file.readlines()
        # checking and removing meta-info
        if lines[3].find('X/Y') == -1:
            print('file ' + filename + ' has wrong format!')
            return None
        line = lines[3].removeprefix('X/Y	').replace(',', '.')
        # getting x coords
        x = [float(n) for n in line.split('\t')]
        y = []
        # making matrix for the input data
        matrix = np.zeros((len(lines) - 4, len(x)))
        for i in range(4, len(lines)):
            # changing ',' delimiter to '.'
            nums = lines[i].replace(',', '.').split('\t')
            # getting current y coord
            y.append(float(nums[0]))
            # filling matrix row
            for j in range(1, len(nums)):
                matrix[i - 4, j - 1] = float(nums[j])
        # swapping x and y because I misunderstood the input format
        x, y = y, x
        matrix = matrix.T
        return x, y, matrix

# getting samples out of files list
def get_data(files_list):
    z_samples = []
    for filename in files_list:
        x, y, z_matrix = read_data(filename)
        # block for drawing examples of input samples
        '''
        grids = np.meshgrid(x, y)
        fig = plot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grids[0], grids[1], z_matrix, cmap='OrRd')
        fig = plot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grids[0], grids[1], median_filter(z_matrix, footprint=np.ones((10, 6)), mode='constant'),
                        cmap='OrRd')
        '''
        # for 2'nd packet, because some samples there have wrong shape
        #if z_matrix.shape[0] * z_matrix.shape[1] == 21717:
        # smoothing the input samples and making common samples list
        z_samples.append(np.ravel(median_filter(z_matrix, footprint=np.ones((10, 6)), mode='constant')))
        # for getting raw samples
        #z_samples.append(np.ravel(z_matrix))
    return x, y, z_samples

# PCA decomposition of the samples represented with res_shape shape
def get_pca_res(samples, res_shape):
    pca = PCA()
    # making long list of the input data
    pca_data = pca.fit_transform(np.array(samples).T.tolist())
    # drawing histogram
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['pc_' + str(x) for x in range(1, len(per_var) + 1)]
    plot.bar(x=range(1, 11), height=per_var[0:10], tick_label=labels[0:10])
    # getting principal components
    pca_df = pd.DataFrame(pca_data, columns=labels)
    xy_grids = np.meshgrid(x, y)
    comp1_grid = np.reshape(pca_df.pc_1.to_numpy(), res_shape)
    comp2_grid = np.reshape(pca_df.pc_2.to_numpy(), res_shape)
    return comp1_grid, comp2_grid

# use to draw 3 dimensional graph
def draw_3d_plot(x_mesh, y_mesh, z_mesh):
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xy_grids[0], xy_grids[1], comp1_grid, cmap='OrRd')
    ax.set_xlabel('Emission, nm.')
    ax.set_ylabel('Excitation, nm.')
    ax.set_zlabel('Intencity')

# use to draw contour graph
def draw_contour(x_mesh, y_mesh, z_mesh):
    fig = plot.figure()
    ax = fig.add_subplot()
    cs = ax.contourf(xy_grids[0], xy_grids[1], comp1_grid, cmap='OrRd', levels=50)
    ax.set_xlabel('Emission, nm.')
    ax.set_ylabel('Excitation, nm.')
    plot.colorbar(cs)

# listing all .txt files in the directory
files_list = filter(lambda str: str.endswith('.txt'), os.listdir('./'))
# getting data from them
x, y, z_samples = get_data(files_list)
# getting x and y grids for graphics
xy_grids = np.meshgrid(x, y)
# getting 1'st and 2'nd principal components
comp1_grid, comp2_grid = get_pca_res(z_samples, xy_grids[0].shape)
# making graphs
draw_3d_plot(xy_grids[0], xy_grids[1], comp1_grid)
draw_contour(xy_grids[0], xy_grids[1], comp1_grid)
plot.show()
