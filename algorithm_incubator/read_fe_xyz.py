import numpy as np
import os
import spectral.io.envi as envi
from spectral import *
import matplotlib
import open3d as o3d
from matplotlib import pyplot as plt

output_dir = '/Users/a1132077/development/hyperspec/point_clouds/'

# file = {
#     'description': 'fx_10',
#     'base_dir': "/Users/a1132077/development/hyperspec/sample_data/ua_mcdonald_2_height_adj/",
#     'header_file': "Combined_SpecimFX10_6_20200928-002.cmb.hdr"
# }

file = {
    'description': 'fx_10',
    'base_dir': "D:/FE_0545_examples",
    'header_file': "Combined_SpecimFX10_NM_3176_20200731-001.cmb.hdr"
}

LiDAR_offset = (2 ** 15 - 1)
normal_offset = (2 ** 15 - 1)
# range is 65k but it's centred about 0 so divide by 32k
normal_extent = (2 ** 15 - 1)

filename = os.path.join(file['base_dir'], file['header_file'])
file['scan'] = envi.open(filename)

LiDAR_bands = np.array([[0, 1], [2, 3], [-2, -1]]) + file['scan'].shape[2] - 11 + 2
LiDAR_normals_bands = np.array([0, 1, 2]) + file['scan'].shape[2] - 11 + 8

points = np.empty(shape=[3, file['scan'].shape[0] * file['scan'].shape[1]])

for b, band in enumerate(LiDAR_bands):
    base_m = file['scan'].read_band(band[0]).astype(np.int32).flatten()
    base_cm = file['scan'].read_band(band[1]).astype(np.int32).flatten()
    shift_m = base_m - LiDAR_offset
    shift_cm = base_cm - LiDAR_offset
    de_normal_m = shift_m / 1
    de_normal_cm = shift_cm / (100 * 100)
    points[b] = np.array([de_normal_m + de_normal_cm])

normals = np.empty(shape=[3, file['scan'].shape[0] * file['scan'].shape[1]])

for b, band in enumerate(LiDAR_normals_bands):
    base_norm = file['scan'].read_band(band).astype(np.int32).flatten()
    base_shift = base_norm - normal_offset
    norm_de_normal = base_shift / normal_extent
    normals[b] = norm_de_normal

file['pcd'] = o3d.geometry.PointCloud()
file['pcd'].points = o3d.utility.Vector3dVector(np.array(points).transpose())
file['pcd'].normals = o3d.utility.Vector3dVector(np.array(normals).transpose())

xyz = np.asarray(file['pcd'].points)
xyz = xyz.reshape((file['scan'].shape[0], file['scan'].shape[1], 3))

f, a = plt.subplots(1, 3)
x = xyz[:, :, 0]
y = xyz[:, :, 1]
z = xyz[:, :, 2]
a[0].imshow(xyz[:, :, 0], cmap='jet')
a[0].set_title('x ' + 'min: ' + str(np.round(np.min(x), 4)) + ' max: ' + str(np.max(x)))
a[1].imshow(xyz[:, :, 1], cmap='jet')
a[1].set_title('y ' + 'min: ' + str(np.round(np.min(y), 4)) + ' max: ' + str(np.max(y)))
a[2].imshow(xyz[:, :, 2], cmap='jet')
a[2].set_title('z ' + 'min: ' + str(np.round(np.min(z), 4)) + ' max: ' + str(np.max(z)))
plt.show()
