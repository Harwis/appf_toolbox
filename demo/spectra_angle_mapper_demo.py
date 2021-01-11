from appf_toolbox.hyper_processing import transformation as tf
import numpy as np
from matplotlib import pyplot as plt

# Parameters
data_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data'
data_name = 'hypcube.npy'
id_band = 100
row_target1 = 214 # Target1 is a spectral signature of the leaf
col_target1 = 252
row_target2 = 297 # Target2 is a spectral signature of the blue marker
col_target2 = 250
member_id = 0
flag_fig = True

# Load a hypercube
hc = np.load(data_path + '/' + data_name)

# Show a image
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.imshow(hc[:, :, id_band])
ax1.scatter(col_target1, row_target1, color='red', marker='+')
ax1.text(col_target1 + 3, row_target1, 'Target 0: leaf', color='red')
ax1.scatter(col_target2, row_target2, color='red', marker='+')
ax1.text(col_target2 + 3, row_target2, 'Target 1: blue marker', color='red')

# Make the members (target spectral signatures)
mem_leaf = hc[row_target1, col_target1, :].reshape((1, hc.shape[2]), order='C')
mem_blue = hc[row_target2, col_target2, :].reshape((1, hc.shape[2]), order='C')
mem = np.concatenate((mem_leaf, mem_blue), axis=0)

# Spectral angles
sa = tf.spectral_angle_mapper(hc, mem, flag_figure=True, member_id=0)




