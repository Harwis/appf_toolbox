import numpy as np
import appf_toolbox.hyper_processing.pre_processing as pp

print(pp.remove_jumps.__doc__)

# Parameters:
data_path = '/media/huajian/Files/python_projects/appf_toolbox_demo_data'
data_name = 'FieldSpec_demo_data.npy'
data_id = 6
ind_jump = [650, 1480] # Manually input the indices of jumps

# Load data
data = np.load(data_path + '/' + data_name, allow_pickle=True)
data = data.flat[0]
ref = data['reflectance']
a_ref = ref[data_id]

# Made fake jump for demo
a_ref[ind_jump[0] + 1:] = a_ref[ind_jump[0] + 1:] + 0.1
a_ref[ind_jump[1] + 1:] = a_ref[ind_jump[1] + 1:] -0.05


ref = pp.remove_jumps(a_ref, ind_jumps=ind_jump, flag_auto_detect_jumps=False, flag_plot_result=True)

# Automatically find the jump
ref = pp.remove_jumps(a_ref, ind_jumps=ind_jump, flag_auto_detect_jumps=True, flag_plot_result=True)