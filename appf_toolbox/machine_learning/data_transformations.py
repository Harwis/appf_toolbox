def hc2hhsi(hc):
    import numpy as np
    # Transform an n-dimensional hypercube to hyper-hue, saturation and intensity.

    # Citation 1:
    # Liu, H., Lee, S., & Chahl, J.(2017).Transformation of a high-dimensional color space for material classification.
    # Journal of the Optical Society of America A, 34(4), 523 - 532, doi: 10.1364/josaa.34.000523.

    # Citation 2:
    # Liu, H., & Chah, J.S.(2018).A multispectral machine vision system for invertebrate detection on green leaves
    # Computer and Elecronics in Agriculture, 150, 279 - 288, doi: https://doi.org/10.1016/j.compag.2018.05.002.

    # Input:  hc: hyperCube(rows x cols x dims) in floating data type of [0 1].
    # Output: hh: hypHue (rows x cols x (dims-1)) in floating data type of [0 1].
    #         s: saturation (rows x cols ) in floating data type of [0 1]
    #         i: intensity in (rows x cols) in floating data type of [0 1]

    # Author: Huajian Liu
    # Date: Aug 18, 2018

    ####################################################################################################################
    # Calculate the components c
    rows = hc.shape[0]
    cols = hc.shape[1]
    dims = hc.shape[2]

    c = np.zeros((rows, cols, dims-1))
    for i in range(dims - 1):
        nonZeroEle = dims - i # nonZeroEle is the number of non-zero elements of the base unit vector u1, u2, ...
        c[:, :, i] = (nonZeroEle - 1) ** 0.5 / nonZeroEle ** 0.5         * hc[:, :, i] \
                     - 1 / ((nonZeroEle - 1) ** 0.5 * nonZeroEle ** 0.5) * np.sum(hc[:, :, i+1:dims], axis=2)
    ####################################################################################################################

    # Normalise the norms of c to 1 to obtain hyper-hue hh.
    c_norm = np.sum(c ** 2, axis=2) ** 0.5
    c_norm = c_norm + (c_norm == 0) * 1e-10
    c_norm = np.tile(c_norm, (dims - 1, 1, 1))
    c_norm = np.moveaxis(c_norm, 0, -1)
    hh = c / c_norm # add 1e-10 to avoid zero denominators

    # Saturation
    s = hc.max(2) - hc.min(2)
    # s = np.amax(hc, axis=2) - np.amin(hc, axis=2) # The same as above

    # Intensity
    i = 1/dims * np.sum(hc, 2)

    return hh, s, i
    # -------------------------------------------------------------------------------

