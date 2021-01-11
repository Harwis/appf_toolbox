def rotate_hypercube(hypercube, angle, scale=1, center='middle', flag_show_img=False, band_check=100):
    """
    Rotate a hypercube.

    :param hypercube: an ndarray of size (rows, cols, bands)
    :param angle: 0-360 degrees
    :param scale: the scale of images; in the range of (0, 1]; defual is 1.
    :param center: the center of the rotation; default is (cols/2, rows/2)
    :param flag_show_img: the flag to show the images or not
    :param band_check: The band at which the image will be checked.
    :return: the rotatd hypercube

    Author: Huajian Liu

    Version: v0 (Nov 25, 2019)
    """
    from matplotlib import pyplot as plt
    import cv2

    cols = hypercube.shape[1]
    rows = hypercube.shape[0]

    if center=='middle':
        # Use default values
        center = (cols/2, rows/2)
    else:
        # Use the (cols, rows) input by user.
        pass

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # warp the image
    dst = cv2.warpAffine(hypercube, rotation_matrix, (cols, rows))

    if flag_show_img:
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(hypercube[:, :, band_check], cmap='gray')
        ax1.set_title('Original image')
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(dst[:, :, band_check], cmap='gray')
        ax2.set_title('Rotated image')

    return dst

if __name__ == '__main__':
    import numpy as np
    hypcube = np.load('demo_data/hypcube.npy')
    rotate_hypercube(hypcube, 180, flag_show_img=True, band_check=10)



