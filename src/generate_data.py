import numpy as np

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import src.zeldovich as Z


def simulation(args):
    _, Ngrid, pkinit, Lbox = args
    ic = np.random.normal(0., 1., (Ngrid, Ngrid, Ngrid))
    final_density, initial_conditions = Z.run_wn(0., ic, pkinit, boxsize=Lbox, ngrid=Ngrid)
    return (initial_conditions, final_density)

def get_simulated_data(n_simulations = 12000,
                       power_spectrum_file_location='colossus_generated_data/pk_my_cosmo.txt', 
                       simulation_box_size = 256.0, 
                       grid_size = 32, 
                       random_seed=123):
    # setting a seed
    np.random.seed(random_seed)
    
    # Set the parameters
    Lbox = simulation_box_size  # Size of the simulation box
    Ngrid = grid_size  # Grid size
    pkfile = power_spectrum_file_location  # File path for the power spectrum

    # Load power spectrum from txt file
    pkinit = np.loadtxt(pkfile)

    num_simulations = n_simulations
    
    with Pool(cpu_count()) as p:
        data = list(tqdm(p.imap(simulation, [(i, Ngrid, pkinit, Lbox) for i in range(num_simulations)], chunksize=1), total=num_simulations))
    
    return data
