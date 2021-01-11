# import numpy as np
# from matplotlib import pyplot as plt
from sklearn.externals import joblib
from appf_toolbox.machine_learning import regression as rg

# Parameters
data_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data'
data_name = 'grass_vnir_n.sav'

# Para for PLSR
max_n_components = 15
num_folds_outer_cv = 3
num_folds_inner_cv = 3

# Load data
ref_n = joblib.load(data_path + '/' + data_name)
ref = ref_n['ref']
lab = ref_n['lab']
wav = ref_n['wavelengths']


record_cv = rg.modelling_PLSRegression(max_n_components,
                                       num_folds_outer_cv,
                                       num_folds_inner_cv,
                                       ref,
                                       wav,
                                       lab,
                                       flag_save=False,
                                       flag_fig=True,
                                       id_cv=2)