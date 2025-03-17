import os
os.environ["OMP_NUM_THREADS"] = "1"

from dolfinx import fem, geometry
from dolfinx.io import gmshio
import ufl
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

import numpy as np
import utils
import global_assembler as ga
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--array', type=int)
args = parser.parse_args()

from pathlib import Path
import yaml
import sys

if __name__ == "__main__":
    if rank == 0:
        print("\nStart post-processing for tsv array {}!".format(args.array))

    with open(args.config, 'r') as file:
            config = yaml.safe_load(file)

    output_dir = Path(config['output_dir'])
    if not output_dir.exists():
        raise ValueError("Invalid path")
    
    materials = config['material']
    E = []
    v = []
    alpha = []
    for i, material in enumerate(materials):
        E.append(materials[i]['young_modulus'])
        v.append(materials[i]['poisson_ratio'])
        alpha.append(materials[i]['thermal_expansion_coefficient'])

    lambda__ = []
    mu_ = []
    alpha_ = []
    for i in range(len(E)):
        lambda__.append(E[i]*v[i]/(1+v[i])/(1-2*v[i]))
        mu_.append(E[i]/2/(1+v[i]))
        alpha_.append(alpha[i]*(3*lambda__[i]+2*mu_[i]))

    thermal_load = config['temperature']

    pitch = config['geometry']['pitch']
    diameter = config['geometry']['diameter']
    height = config['geometry']['height']
    
    interp_num = config['solver']['interp_num']
    block_tsv_num = config['solver']['block_tsv_num']

    tsv_num = config['tsv_array'][args.array]['tsv_num']
    block_num = {'x':tsv_num['x']//block_tsv_num['x'], 'y':tsv_num['y']//block_tsv_num['y']}

    dummy_tsv_num = config['tsv_array'][args.array]['dummy_tsv_num']
    dummy_block_num = {'x':dummy_tsv_num['x']//block_tsv_num['x'], 'y':dummy_tsv_num['y']//block_tsv_num['y']}

    grid_num = config['plot']
    nx, ny, nz = (grid_num['x'], grid_num['y'], grid_num['z'])
    lx, ly, lz = (pitch*block_tsv_num['x'], pitch*block_tsv_num['y'], height)
    interval_x = lx/nx; interval_y = ly/ny; interval_z = lz/nz
    x_ = np.arange(0.5, 0.5+nx)*interval_x-lx/2; y_ = np.arange(0.5, 0.5+ny)*interval_y-ly/2; z_ = np.arange(0.5, 0.5+nz)*interval_z-lz/2
    xx, yy, zz = np.meshgrid(x_, y_, z_)
    local_grid_points = np.vstack((xx.reshape((1, -1)), yy.reshape((1, -1)), zz.reshape((1, -1))))

    dummy_tag = 0
    if dummy_block_num['x'] > 0 or dummy_block_num['y'] > 0:
        dummy_tag = 1
    
    x = np.load(output_dir/('x'+str(args.array)+'.npy'))
    modes = np.load(output_dir/"modes0.npy")
    modes_dummy = None
    if dummy_tag:
        modes_dummy = np.load(output_dir/"modes1.npy")
    
    if comm.rank == 0:
        domain, ct, _ = gmshio.read_from_msh(str(output_dir/'mesh0.msh'), MPI.COMM_SELF, 0, gdim=3)
    else:
        with utils.suppress_stdout_stderr():
            domain, ct, _ = gmshio.read_from_msh(str(output_dir/'mesh0.msh'), MPI.COMM_SELF, 0, gdim=3)
    sys.stdout.reconfigure(line_buffering=True)

    V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    uh = fem.Function(V)
    lambda_, mu, alpha = utils.assign_materials(domain, list(range(1, len(E)+1)), [lambda__, mu_, alpha_], ct)
    temperature = fem.Constant(domain, thermal_load)

    V_von_mises = fem.functionspace(domain, ("DG", 2))
    stresses = fem.Function(V_von_mises)

    bb_tree = geometry.bb_tree(domain, domain.geometry.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, local_grid_points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, local_grid_points.T)
    local_cells = [colliding_cells.links(i)[0] for i,_ in enumerate(local_grid_points.T)]

    uh_dummy = None
    stresses_dummy = None
    local_cells_dummy = None
    V_von_mises_dummy = None
    if dummy_tag:
        if comm.rank == 0:
            domain, ct, _ = gmshio.read_from_msh(str(output_dir/'mesh1.msh'), MPI.COMM_SELF, 0, gdim=3)
        else:
            with utils.suppress_stdout_stderr():
                domain, ct, _ = gmshio.read_from_msh(str(output_dir/'mesh1.msh'), MPI.COMM_SELF, 0, gdim=3)
        sys.stdout.reconfigure(line_buffering=True)
        V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
        uh_dummy = fem.Function(V)
        lambda_dummy, mu_dummy, alpha_dummy = utils.assign_materials(domain, list(range(1, len(E)+1)), [lambda__, mu_, alpha_], ct)
        temperature_dummy = fem.Constant(domain, thermal_load)

        V_von_mises_dummy = fem.functionspace(domain, ("DG", 2))
        stresses_dummy = fem.Function(V_von_mises_dummy)

        bb_tree = geometry.bb_tree(domain, domain.geometry.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, local_grid_points.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, local_grid_points.T)
        local_cells_dummy = [colliding_cells.links(i)[0] for i,_ in enumerate(local_grid_points.T)]
        

    global_element_dofs = [(i, j, k, d) 
                           for i in range(interp_num['x']) 
                           for j in range(interp_num['y']) 
                           for k in range(interp_num['z']) 
                           for d in range(3)
                           if i*j*k==0 or i==interp_num['x']-1 or j==interp_num['y']-1 or k==interp_num['z']-1]
    local2global, _= ga.create_local2global_mapping(global_element_dofs, block_num['x']+2*dummy_block_num['x'], block_num['y']+2*dummy_block_num['y'], interp_num['x'], interp_num['y'])
    dummy_tags = ga.locate_dummy_blocks(block_num['x'], block_num['y'], dummy_block_num['x'], dummy_block_num['y'])

    rank_blocks = utils.get_rank_parts(np.arange((block_num['x']+2*dummy_block_num['x'])*(block_num['y']+2*dummy_block_num['y'])), rank, size)
    rank_values = np.empty((len(rank_blocks), local_grid_points.shape[1]), dtype=np.float64)
    for i, block in enumerate(rank_blocks):
        if dummy_tags[block] == 0:
            uh.x.array[:] = x[local2global[block]]@modes[:-1]+thermal_load*modes[-1]
            stress_expr = fem.Expression(utils.von_mises(uh, lambda_, mu, alpha, temperature), V_von_mises.element.interpolation_points())
            stresses.interpolate(stress_expr)
            rank_values[i] = np.linalg.norm(stresses.eval(local_grid_points.T, local_cells), axis=1, keepdims=False)
        else:
            uh_dummy.x.array[:] = x[local2global[block]]@modes_dummy[:-1]+thermal_load*modes_dummy[-1]
            stress_expr = fem.Expression(utils.von_mises(uh_dummy, lambda_dummy, mu_dummy, alpha_dummy, temperature_dummy), V_von_mises_dummy.element.interpolation_points())
            stresses_dummy.interpolate(stress_expr)
            rank_values[i] = np.linalg.norm(stresses_dummy.eval(local_grid_points.T, local_cells_dummy), axis=1, keepdims=False)
    
    values = None
    recv_counts = None
    displaments = None
    rank_blocks = comm.gather(len(rank_blocks), root=0)
    if rank == 0:
        values = np.empty(((block_num['x']+2*dummy_block_num['x'])*(block_num['y']+2*dummy_block_num['y']), local_grid_points.shape[1]), dtype=np.float64).flatten()
        recv_counts = np.array(rank_blocks, dtype=np.int32)*local_grid_points.shape[1]
        displaments = np.insert(np.cumsum(recv_counts[:-1]), 0, 0)
    comm.Gatherv(rank_values.flatten(), [values, recv_counts, displaments, MPI.DOUBLE], root=0)

    if rank == 0:
        values = values.reshape((-1, local_grid_points.shape[1]))
        contour = utils.reshape_to_contour(values, block_num['x']+2*dummy_block_num['x'], block_num['y']+2*dummy_block_num['y'], grid_num['x'], grid_num['y'], grid_num['z'])
        np.save(output_dir/('vonmises'+str(args.array)), contour)

