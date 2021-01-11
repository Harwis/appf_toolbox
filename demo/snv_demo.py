import numpy as np
from matplotlib import pyplot as plt
from appf_toolbox.hyper_processing import transformation as tf

data_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data'
data_name = 'FieldSpec_demo_data.npy'

data = np.load(data_path + '/' + data_name, allow_pickle='True')
data = data.flat[0]
ref = data['reflectance']

snv_data = tf.snv(ref, flag_fig=True)