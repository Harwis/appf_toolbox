def remove_jumps(spec_sig, ind_jumps=[], num_points=5, flag_auto_detect_jumps=False, t_der=0.004, r_win=15,
                 flag_plot_result=False):
    """
    Remove the jumps in a spectral signature.

    :param spec_sig: the spectral signature need to be fixed;1D ndarray format

    :param ind_jumps: the indices of jumps; must be a list; the indices starts from 0; default is []; [650, 1480] for
           FieldSpec data; [398] for hyp images of vnir + swir

    :param  num_points: the number points to fit a line on the left-side of the jumps

    :param flag_auto_detect_jumps: the flag to determine if use auto detection of the jumps or not; default is False

    :param t_der: the threshold of the second derivative; if a value > t_der, then it is a jump; only used when
           flag_auto_detect-jumps == True; default is 0.004 for FieldSpec data; recommend 0.01 for imaging data.
                  
    :param r_win: the radius of the window to distinguish jumps and randoms noises;
           only used when flag_auto_detect_jumps == True; default is 15.
           
    :param flag_plot_result: the flag to indicate if show the results in a figure or not; default is False
    
    :return: If ind_jumps is not empty, return the fixed spectral signature without jumps; otherwise return -1


    Author: Huajian Liu

    Version: v0 (20 Dec, 2019)
    """
    
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.optimize import curve_fit


    # Show a data point to check
    if flag_plot_result:
        fig = plt.figure()
        ax_raw = fig.add_subplot(3, 1, 1)
        ax_raw.plot(spec_sig)
        ax_raw.set_ylabel('Raw data', fontsize=12, fontweight='bold')


    ####################################################################################################################
    # Using the second-order derivative to automatically find the the jumps
    # der of f = (f(x + delt x) - f(x)) / delt x
    # Der of ref(0) is retained and the ref(n) is discarded
    ####################################################################################################################
    if flag_auto_detect_jumps:
        # First order derivative
        fir_der = np.delete(spec_sig, 0) - np.delete(spec_sig, spec_sig.shape[0] - 1)

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
            ax_2der.hlines(t_der, 0, sec_der.shape[0] - 1, colors='red')
            ax_2der.text(0, t_der, 'Threshold')
            # ax_2der.set_title('second order derivative')

        # In values of the second order derivative, tow criterias are used to determine if there is a jump at the index 
        # i. 1; t_der # 2: The number of high-values in the window of [i - r_win, i + r_win + 1] at i. For a real jump, 
        # num_high = 2 
        # First, zero-padding the borders. 
        sec_der_pad = np.concatenate((np.zeros((r_win,)), sec_der, np.zeros((r_win,))))

        # Find the index of the jumps 
        ind_jumps = []
        for i in range(sec_der.shape[0]):
            num_high_in_win = np.sum(np.abs(sec_der_pad[i:i + 2 * r_win + 1]) > t_der)
            if np.logical_and(np.abs(sec_der[i]) > t_der,  num_high_in_win == 2):
                ind_jumps.append(i)

        if ind_jumps != []:
            # The high-values of 2end deri appears as paris. In each pair, remove the first one since only the second
            # one indicate the location of the jump.
            ind_jumps = np.delete(ind_jumps, range(0, ind_jumps.__len__(), 2))  
    else: # Use the manually inputted band numbers. 
        pass


    ####################################################################################################################
    # Fix the jumps
    ####################################################################################################################
    def fit_a_line(x, a, b):
        return a*x + b

    if ind_jumps == []:
        print('No jumps found!')
        return -1
    else:
        # Fix the jump
        est_values = []
        for i_jump in ind_jumps:
            # Fit a line
            x1 = i_jump - num_points
            x2 = i_jump + 1
            popt, pcov = curve_fit(fit_a_line, np.arange(x1, x2),
                                   spec_sig[x1:x2])
            a = popt[0]
            b = popt[1]

            # Using the line to estimate y2. (x1, y1) is the first point for fitting the line. (x2, y2) is the first
            # point after the jump.
            y2 = fit_a_line(i_jump + 1, a, b)
            est_values.append(y2)

            if flag_plot_result:
                y1 = fit_a_line(x1, a, b)
                ax_raw.plot((x1, x2), (y1, y2), 'r--', linewidth=2)
                ax_raw.text(x2, y2-0.05, 'jump', fontsize=12)


        # Estimate the value after the jumps.
        ind_jumps = np.asarray(ind_jumps)
        est_values = np.asarray(est_values)
        raw_values = spec_sig[ind_jumps + 1]
        rel_shifts = raw_values - est_values
        abs_shifts = np.cumsum(rel_shifts)

        # Shit the data
        spec_sig_fixed = spec_sig.copy()
        for i in range(0, ind_jumps.shape[0]):
            if i < ind_jumps.shape[0] - 1:
                spec_sig_fixed[ind_jumps[i] + 1:ind_jumps[i+1] + 1] = \
                spec_sig[ind_jumps[i] + 1:ind_jumps[i+1] + 1] - abs_shifts[i]
            else:
                spec_sig_fixed[ind_jumps[i] + 1:] = spec_sig[ind_jumps[i] + 1:] - abs_shifts[i]

        if flag_plot_result:
            ax_raw.plot(spec_sig_fixed, 'g--')

        return spec_sig_fixed


if __name__ == '__main__':
    import numpy as np
    data_path = '/media/huajian/Files/p_0438_wheatWithMarkers/Python_codes/data'
    data_name = 'sample_ref_HI_full.npy'
    # data_name = 'FieldSpec_demo_data.npy'
    data_id = 14
    ref = np.load(data_path + '/' + data_name)
    a_ref = ref[data_id]
    ind_jump = [398]
    remove_jumps(a_ref, ind_jumps=ind_jump, flag_auto_detect_jumps=True, t_der=0.01, flag_plot_result=True)



