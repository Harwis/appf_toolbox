import numpy as np
from appf_toolbox.hyper_processing import pre_processing as pp

folder_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data'

hypcube = np.load(folder_path + '/hypcube.npy')
hypcube_rotated = pp.rotate_hypercube(hypcube, 180, flag_show_img=True, band_check=100)