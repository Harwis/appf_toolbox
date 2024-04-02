import sys
sys.path.append('../appf_toolbox')
from appf_toolbox.wiwam import wiwam_tools
import spectral.io.envi as envi
from matplotlib import pyplot as plt
import numpy as np

# Parameters
# Data with dark ref errors
# data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/wiw_20210421'
# data_name = 'vnir_92_130_8079_2021-04-21_05-20-00'
# data_name = 'vnir_92_130_8082_2021-04-21_05-21-41'

# data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/wiw_20210428'
# data_name = 'vnir_92_130_8073_2021-04-28_04-00-32'

# data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/wiw_20210506'
# data_name = 'vnir_92_130_8077_2021-05-06_03-43-23'

# data_path = '/media/huajian/Files/Data/error_data/VNIR_dark_error_example'
# data_name = 'vnir_43_54_1790_2018-10-01_22-50-58'
# data_name = 'vnir_74_104_6114_2019-10-31_00-39-51'


# Data without dark ref errors
# data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/wiw_20210421'
# data_name = 'vnir_92_130_8076_2021-04-21_05-53-27'
# data_name = 'vnir_92_130_8080_2021-04-21_05-25-02'

# data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/wiw_20210506'
# data_name = 'vnir_92_130_8063_2021-05-06_03-55-10'

# data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/wiw_20210520'
# data_name = 'vnir_92_130_8072_2021-05-20_07-03-43'

data_path = '/media/huajian/Files/Data/wheat_gene_classification_0528'
data_name = 'vnir_86_121_7559_2020-08-17_03-47-26'

check_band = 300
check_row_col = [50, 250]

n_bins = 20

# Load data
meta_data = envi.open(data_path + '/' + data_name + '/capture/' + 'DARKREF_' + data_name + '.hdr',
                      data_path + '/' + data_name + '/capture/' + 'DARKREF_' + data_name + '.raw')

data = meta_data.load()
wave = np.asarray(data.metadata['wavelength'], dtype='float')

# min and max
min = data.min()
max = data.max()

# Plot
f0, ax_f0 = plt.subplots(2, 2)
f0.suptitle(data_name)
ax_f0[0, 0].imshow(data[:, :, check_band], cmap='jet')
ax_f0[0, 0].scatter(check_row_col[1], check_row_col[0], marker='o', color='black', s=30)
ax_f0[0, 0].scatter(check_row_col[1], check_row_col[0], marker='+', color='white', s=25)
ax_f0[0, 0].set_title('Band ' + str(check_band) + '@' + str(wave[check_band]) + 'nm')
ax_f0[0, 1].hist(data.reshape((data.shape[0] * data.shape[1] * data.shape[2], )), bins=n_bins)
ax_f0[0, 1].set_title('min: %.2f max: %.2f' % (min, max))

for ax in ax_f0[1, : ]:
    ax.remove()
gs = ax_f0[1, 1].get_gridspec()
ax_f0_34 = f0.add_subplot(gs[1, :])

ax_f0_34.plot(wave, data[check_row_col[0], check_row_col[1], :].reshape((data.shape[2], )))
ax_f0_34.set_xlabel('Wavelength (nm)')
ax_f0_34.set_ylabel('Raw values')
plt.show()

# Check the check_fx10_dark_error()
check_result = wiwam_tools.check_fx10_dark_error(data_path, data_name)
if check_result == 1:
    print(data_name + ' has errors.')
else:
    print(data_name + ' does not have errors')

