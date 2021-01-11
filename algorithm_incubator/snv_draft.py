import numpy as np
from matplotlib import pyplot as plt

data_path = 'demo_data'
data_name = 'FieldSpec_demo_data.npy'

data = np.load(data_path + '/' + data_name)
data = data.flat[0]
ref = data['reflectance']
wave = data['wavelength']
print('The shape of ref', ref.shape)

mean_ref = np.mean(ref, axis=1)
std_ref = np.std(ref, axis=1)

mean_ref = mean_ref.reshape((mean_ref.shape[0], 1), order='C')
std_ref = std_ref.reshape((std_ref.shape[0], 1), order='C')

mean_ref = np.tile(mean_ref, (1, ref.shape[1]))
std_ref = np.tile(std_ref, (1, ref.shape[1]))

snv = (ref - mean_ref) / std_ref

f = plt.figure()
a1 = f.add_subplot(1, 2, 1)
a2 = f.add_subplot(1, 2, 2)

for a_ref in ref:
    a1.plot(wave, a_ref)

for a_snv in snv:
    a2.plot(wave, a_snv)