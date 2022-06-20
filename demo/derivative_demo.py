import sys
sys.path.append('E:/python_projects/appf_toolbox_project')
import numpy as np
import appf_toolbox.hyper_processing.transformation as tr

# Parameters:
# data_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data'
data_path = 'E:/python_projects/appf_toolbox_demo_data'

data_name = 'FieldSpec_demo_data.npy'
data_id = 6
ind_jump = [650, 1480] # Manually input the indices of jumps

# Load data
data = np.load(data_path + '/' + data_name, allow_pickle=True)
data = data.flat[0]
ref = data['reflectance']
a_ref = ref[data_id]

# 1st-order derivative

a_first_dev = tr.first_order_derivative(a_ref, flag_check=True)
first_dev = tr.first_order_derivative(ref, flag_check=True, check_interview=2)

a_first_dev = tr.second_order_derivative(a_ref, flag_check=True)
first_dev = tr.second_order_derivative(ref, flag_check=True, check_interview=2)
