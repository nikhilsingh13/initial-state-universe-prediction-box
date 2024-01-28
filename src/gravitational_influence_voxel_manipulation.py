import numpy as np

def remove_center_and_fill(data, fill_value=0):
    """
    This function modifies a 3D numpy array (representing a voxel grid) by removing a 
    central portion and filling it with a specified value. It is designed for use in 
    astrophysical simulations to test the hypothesis that surrounding voxel values can 
    extrapolate missing information due to the effects of gravitational nuances, which 
    are more pronounced at shorter distances compared to influences like dark energy.

    Parameters:
    data (numpy.ndarray): A 3D numpy array representing the voxel grid from which 
                          the central portion will be removed.
    fill_value (int, optional): The value with which the removed central portion 
                                will be filled. Defaults to 0.

    Returns:
    numpy.ndarray: The modified 3D numpy array with the central portion removed 
                   and filled with `fill_value`.

    The function calculates the center of the given 3D array and identifies a smaller 
    cube within this central region. The values within this smaller cube are then set 
    to `fill_value`. This manipulation simulates a scenario where central information 
    is missing, allowing for the testing of models' ability to predict these missing 
    values based on the surrounding data, thereby simulating gravitational effects 
    within the universe.
    """
    
    modified_data = np.copy(data)
    center = data.shape[0] // 2  # for a cubic grid
    small_cube_size = center // 2  # smaller cube

    # Calculate start and end indices for the smaller cube
    start = center - small_cube_size // 2
    end = center + small_cube_size // 2

    # Set the values in the smaller cube to fill_value
    modified_data[start:end, start:end, start:end] = fill_value

    return modified_data