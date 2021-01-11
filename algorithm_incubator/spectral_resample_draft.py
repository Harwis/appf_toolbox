from spectral import *
import numpy as np
from matplotlib import pyplot as plt

print(BandResampler.__doc__)

data_path = 'demo_data'
data_name = 'spec_samples_raw.npy' # 14
data_id = 14

ref = np.load(data_path + '/' + data_name)
a_ref = ref[data_id]
wavelength = [i for i in range(350, 2501)]
wavelength = np.asarray(wavelength)

new_wavelength = [i for i in range(350, 2501, 50)]
new_wavelength = np.asarray(new_wavelength)

resampler = BandResampler(wavelength, new_wavelength)
ref_new = resampler(a_ref)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(wavelength, a_ref)
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(new_wavelength, a_ref_new)