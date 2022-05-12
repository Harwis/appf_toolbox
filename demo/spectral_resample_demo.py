from spectral import *
import numpy as np
from matplotlib import pyplot as plt
import appf_toolbox.hyper_processing.pre_processing as pp

print(BandResampler.__doc__)

# data_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data'
data_path = 'E:/python_projects/appf_toolbox_demo_data'
data_name = 'FieldSpec_demo_data.npy'
data_id = 1

data = np.load(data_path + '/' + data_name, allow_pickle=True)
data = data.flat[0]
ref = data['reflectance']
wavelength = data['wavelength']
new_wavelength = [i for i in range(wavelength[0], wavelength[-1], 200)]
new_wavelength = np.asarray(new_wavelength)

a_ref = ref[data_id]
new_ref = pp.spectral_resample(a_ref, wavelength, new_wavelength, flag_fig=True, id_check=10)
