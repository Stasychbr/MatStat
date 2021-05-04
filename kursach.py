import numpy as np
import matplotlib.pyplot as plot
import os


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
        return x, y, matrix

files_list = filter(lambda str: str.endswith('.txt'), os.listdir('./'))
for filename in files_list:
    x, y, z_matrix = read_data(filename)
    grids = np.meshgrid(x, y)
    fig = plot.figure()
    ax = fig.add_subplot()
    ax.contour(grids[0], grids[1], z_matrix, cmap='viridis', levels=30)
    ax.set_title(filename)
plot.show()
