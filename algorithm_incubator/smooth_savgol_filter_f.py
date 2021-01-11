def smooth_savgol_filter_f(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0,
                           flag_fig = False, id_x=0):
    """
    Smooth curves using savgol filter. Call scipy.signal.savgol_filter.

    :param x: Can be 1D or 2D ndarray
    :param window_length:
    :param polyorder:
    :param deriv:
    :param delta:
    :param axis:
    :param mode:
    :param cval:
    :param flag_fig:
    :param id_x:
    :return:

    Author: Huajian Liu
    Email: huajian.liu@adelaide.edu.au

    Version: v0 (29, Nov, 2019)
    """

    # Call scipy.singal.savgol_filter
    from scipy import signal
    x_sm = signal.savgol_filter(x, window_length, polyorder, deriv, delta, axis, mode, cval)

    # Pick up a point to check
    if flag_fig:
        if x.shape.__len__() == 1:
            a_point = x
            a_point_sm = x_sm
        else:
            a_point = x[id_x]
            a_point_sm = x_sm[id_x]

        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(a_point, 'g', linewidth=3, label='Original')
        ax.plot(a_point_sm, 'r--', label='Smoothed')
        ax.set_title('Smooting using savgol filter')
        ax.legend()

    return x_sm


if __name__ == '__main__':
    print(smooth_savgol_filter_f.__doc__)

    from sklearn.externals import joblib

    data_path = 'demo_data'
    data_name = 'grass_vnir_n.sav'
    data_id = 10
    window_length = 11
    polyorder = 3
    id_data = 100

    ref_n = joblib.load(data_path + '/' + data_name)
    ref = ref_n['ref']
    data_sm = smooth_savgol_filter_f(ref, window_length, polyorder, flag_fig=True, id_x=id_data)



