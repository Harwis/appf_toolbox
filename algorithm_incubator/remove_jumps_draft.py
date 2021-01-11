import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

########################################################################################################################
# Parameters
########################################################################################################################
data_path = '/media/huajian/Files/p_0438_wheatWithMarkers/Python_codes/data'
data_name = 'sample_ref_HI_full.npy'
data_name = 'spec_samples_raw.npy'
data_id = 82 # 17 demo of first order not work

noisy_wave_start = 350
noisy_save_stop = 400 # good bands include 400
t_der = 0.004 # 0.004
flag_remove_noisy_band = False
r_win = 15
num_points = 5 # The number of points to fit a line

flag_auto_detect_jumps = False
ind_iumps = [650, 1480] # The band number is 0, 1, ...; For data of FieldSpec [650 1480]; For vnir + swir [398]
flag_plot_result = True

########################################################################################################################
# Load the reflectance data
########################################################################################################################
ref = np.load(data_path + '/' + data_name)

# Remove noise band
if flag_remove_noisy_band:
    wavelengths = np.load('data/spec_wavelengths_raw.npy')
    noisy_band_start = np.where(wavelengths == noisy_wave_start)[0][0]
    noisy_band_stop = np.where(wavelengths == noisy_save_stop)[0][0]
    ref = np.delete(ref, np.arange(noisy_band_start, noisy_band_stop), axis=1)

# Show a ref data point to check
a_ref = ref[data_id]

if flag_plot_result:
    fig = plt.figure()
    ax_raw = fig.add_subplot(3, 1, 1)
    ax_raw.plot(a_ref)
    ax_raw.set_ylabel('Raw ref', fontsize=12, fontweight='bold')


########################################################################################################################
# Using the second-order derivative to automatically find the the jump
# der of f = (f(x + delt x) - f(x)) / delt x
# Der of ref(0) is retained and the ref(n) is discarded
########################################################################################################################
if flag_auto_detect_jumps:
    # First order derivative
    fir_der = np.delete(a_ref, 0) - np.delete(a_ref, a_ref.shape[0] - 1)

    if flag_plot_result:
        ax_1der = fig.add_subplot(3, 1, 2)
        ax_1der.plot(fir_der)
        ax_1der.set_xlabel('band number', fontsize=12, fontweight='bold')
        ax_1der.set_ylabel('1st der', fontsize=12, fontweight='bold')
        # ax_1der.set_title('First order derivative')

    # Second order derivative
    sec_der = np.delete(fir_der, 0) - np.delete(fir_der, fir_der.shape[0] - 1)

    if flag_plot_result:
        ax_2der = fig.add_subplot(3, 1, 3)
        ax_2der.plot(sec_der)
        ax_2der.set_xlabel('band number', fontsize=12, fontweight='bold')
        ax_2der.set_ylabel('2end der', fontsize=12, fontweight='bold')
        # ax_2der.set_title('second order derivative')

    # The algorithm only detect jumps due to the change of sensors; not for random noises.
    # If a jump has more than two neighbouring jumps in the radius r_win, it is considered as a noise and not processed.
    # First, zero padding for slicing window
    sec_der_pad = np.concatenate((np.zeros((r_win,)), sec_der, np.zeros((r_win,))))

    # Find the index of the jumps
    # In the window of [i - r_win, i + r_win + 1], the number of 'real' jumps is two.
    ind_iumps = []
    for i in range(sec_der.shape[0]):
        num_jumps_in_win = np.sum(np.abs(sec_der_pad[i:i + 2 * r_win + 1]) > t_der)
        if np.logical_and(np.abs(sec_der[i]) > t_der,  num_jumps_in_win == 2):
            ind_iumps.append(i)

    if ind_iumps != []:
        # The jumps of 2end deri appears as paris. In each pair, remove the first one since the only the second one
        #  indicate
        # the location of jump.
        ind_iumps = np.delete(ind_iumps, range(0, ind_iumps.__len__(), 2))  # ind_iumps is casted to array automatically.
else: # Use the band numbers of jumps manually inputted
    pass


########################################################################################################################
# Fix the jumps
########################################################################################################################
def fit_a_line(x, a, b):
    return a*x + b

if ind_iumps == []:
    print('No jumps found!')
else:
    # Fix the jump
    est_values = []
    for i_jump in ind_iumps:
        # Fit a line
        popt, pcov = curve_fit(fit_a_line, np.arange(i_jump - num_points, i_jump + 1), a_ref[i_jump - num_points:i_jump + 1])

        # Using the line to estimate y2. (x1, y1) is the first point for fitting the line. (x2, y2) is the first point
        # after the jump.
        y2 = fit_a_line(i_jump + 1, popt[0], popt[1])
        est_values.append(y2)

        if flag_plot_result:
            x1 = i_jump - num_points
            x2 = i_jump + 1
            y1 = fit_a_line(i_jump - num_points, popt[0], popt[1])
            ax_raw.plot((x1, x2), (y1, y2), 'r--', linewidth=2)
            ax_raw.text(x2, y2-0.05, 'jump', fontsize=12)


    # Estimate the value after the jumps.
    ind_iumps = np.asarray(ind_iumps)
    est_values = np.asarray(est_values)
    raw_values = a_ref[ind_iumps + 1]
    rel_shifts = raw_values - est_values
    abs_shifts = np.cumsum(rel_shifts)

    # Shit the data
    a_ref_fixed = a_ref.copy()
    for i in range(0, ind_iumps.shape[0]):
        if i < ind_iumps.shape[0] - 1:
            a_ref_fixed[ind_iumps[i] + 1:ind_iumps[i+1] + 1] = a_ref[ind_iumps[i] + 1:ind_iumps[i+1] + 1] - abs_shifts[i]
        else:
            a_ref_fixed[ind_iumps[i] + 1:] = a_ref[ind_iumps[i] + 1:] - abs_shifts[i]

    if flag_plot_result:
        ax_raw.plot(a_ref_fixed, 'g--')
