# This program demostrates how to read a hypercube using the spectral toolbox

from spectral import *
from matplotlib import pyplot as plt

# Parameters
path = 'E:/Data/test_shutter/20210531'
folder_name = 'vnir_nil_2021-05-31_00-50-20'
hdr_name = 'DARKREF_vnir_nil_2021-05-31_00-50-20.hdr'
raw_name = 'DARKREF_vnir_nil_2021-05-31_00-50-20.raw'

# Reading data
meta_dark = envi.open(path + '/' + folder_name + '/' + 'capture' + '/' + hdr_name,
                       path + '/' + folder_name + '/' + 'capture' + '/' + raw_name)

dark = meta_dark.load()
wave = meta_dark.metadata['wavelength']

plt.figure()
for i in range(0, wave.__len__(), 20):
    plt.imshow(dark[:, :, i], cmap='gray')
    plt.title(wave[i] + 'nm')
    plt.pause(3)

print('Done')

