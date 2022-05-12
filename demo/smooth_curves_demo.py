from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from appf_toolbox.hyper_processing import pre_processing as pp


# Parameters
data_path = '../appf_toolbox_demo_data'
data_name = 'grass_vnir_n.sav'
data_id = 10
window_length = 11
polyorder = 3
lmbda = 5

# Load data
ref_n = joblib.load(data_path + '/' + data_name)
ref = ref_n['ref']
a_ref = ref[data_id]
wl = ref_n['wavelengths']



########################################################################################################################
# Compare savgol filter and spline
########################################################################################################################
# Option 1: savgol filter
# https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.savgol_filter.html
a_ref_savgol = signal.savgol_filter(ref, window_length=window_length, polyorder=polyorder)[data_id]

ref_sm = pp.smooth_savgol_filter(ref, window_length, polyorder, flag_fig=True, id_x=10)

# Smooting could make the value out of the range of [0, 1]. Make the values of reflectance to [0, 1]
ref_sm[ref_sm < 0] = 0
ref_sm[ref_sm > 1] = 1

# Option 2: spline
# Input must be a 2D matrix?
a_ref_spline = signal.spline_filter(ref, lmbda=lmbda)[data_id]

# Plot
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(wl, a_ref, 'r', label='Original data')
ax1.plot(wl, a_ref_savgol, 'g--', label='Smoothed using savgol')
ax1.legend()
ax1.set_xlabel('Wavelengths', fontsize=12, fontweight='bold')
ax1.set_ylabel('Reflectance', fontsize=12, fontweight='bold')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(wl, a_ref, 'r', label='Original data')
ax2.plot(wl, a_ref_spline, 'g--', label='Smoothed using spline')
ax2.legend()
ax2.set_xlabel('Wavelengths', fontsize=12, fontweight='bold')
ax2.set_ylabel('Reflectance', fontsize=12, fontweight='bold')






