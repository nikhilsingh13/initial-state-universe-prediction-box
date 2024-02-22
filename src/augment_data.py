from scipy.ndimage import rotate, shift
import numpy as np

def augment_data_3d_voxel_shift(data, shift_range=1, rotate_voxel=True):
    """
    Augment data by shifting a line of pixels in the voxel and rotating the voxel.
    
    Parameters:
    - data: 3D numpy array, basically voxel.
    - shift_range: for shifting pixels. default = 1.
    - rotate_voxel: boolean, default = True. If True, the voxel will be rotated.
    
    Returns:
    - Augmented 3D numpy array.
    """
    # Randomly choosing shifting values within the given range for each axis
    shift_values = np.random.randint(-shift_range, shift_range + 1, 3)
    
    # Shift the voxel
    shifted_data = shift(data, shift_values, mode='nearest')
    
    if rotate_voxel:
        # Randomly choosing rotation angles for each axis
        angle_x, angle_y, angle_z = np.random.uniform(0, 360), np.random.uniform(0, 360), np.random.uniform(0, 360)
        
        shifted_data = rotate(shifted_data, angle_x, axes=(0, 1), reshape=False)
        shifted_data = rotate(shifted_data, angle_y, axes=(1, 2), reshape=False)
        shifted_data = rotate(shifted_data, angle_z, axes=(0, 2), reshape=False)
    
    return shifted_data
