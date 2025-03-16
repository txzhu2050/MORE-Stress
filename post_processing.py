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
parser.add_argument('-sol_fname', '--solution_filename', type=str)
parser.add_argument('-rst_fname', '--result_filename', type=str)
parser.add_argument('-dir', '--output_dir', type=str)
parser.add_argument('-E', '--young_modulus', nargs='+', type=float)
parser.add_argument('-v', '--poisson_ratio', nargs='+', type=float)
parser.add_argument('-alpha', '--thermal_expansion_coefficient', nargs='+', type=float)
parser.add_argument('-T', '--thermal_load', type=float)
parser.add_argument('-p', '--pitch', type=float)
parser.add_argument('-ht', '--height', type=float)
parser.add_argument('-nt', '--local_tsv_num', nargs='+', type=int)
parser.add_argument('-ni', '--interp_num', nargs='+', type=int)
parser.add_argument('-nb', '--block_num', nargs='+', type=int)
parser.add_argument('-nd', '--dummy_block_num', nargs='+', type=int)
parser.add_argument('-ng', '--grid_num', nargs='+', type=int)
args = parser.parse_args()
from pathlib import Path
import sys

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u, lambda_, mu, alpha, temperature):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u) - alpha * ufl.Identity(3) * temperature

def von_mises(u, lambda_, mu, alpha, temperature):
    s = sigma(u, lambda_, mu, alpha, temperature) - 1. / 3 * ufl.tr(sigma(u, lambda_, mu, alpha, temperature)) * ufl.Identity(len(u))
    return ufl.sqrt(3. / 2 * ufl.inner(s, s))

if __name__ == "__main__":
    young_modulus = args.young_modulus
    poisson_ratio = args.poisson_ratio
    thermal_expansion_coefficient = args.thermal_expansion_coefficient
    thermal_load = args.thermal_load
    pitch = args.pitch
    height = args.height
    local_tsv_num = args.local_tsv_num
    interp_num = args.interp_num
    block_num = args.block_num
    dummy_block_num = args.dummy_block_num
    grid_num = args.grid_num
    sol_fname = args.solution_filename
    rst_fname = args.result_filename
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        raise ValueError("Invalid path")
    
    dummy_tag = 0
    if dummy_block_num[0] > 0 or dummy_block_num[1] > 0:
        dummy_tag = 1
    x = np.load(output_dir/(sol_fname+'.npy'))
    modes = np.load(output_dir/"modes0.npy")
    modes_dummy = None
    if dummy_tag:
        modes_dummy = np.load(output_dir/"modes1.npy")
    
    lambda__ = []
    mu_ = []
    alpha_ = []
    for i in range(len(young_modulus)):
        lambda__.append(young_modulus[i]*poisson_ratio[i]/(1+poisson_ratio[i])/(1-2*poisson_ratio[i]))
        mu_.append(young_modulus[i]/2/(1+poisson_ratio[i]))
        alpha_.append(thermal_expansion_coefficient[i]*(3*lambda__[i]+2*mu_[i]))

    nx, ny, nz = grid_num; lx, ly, lz = (pitch*local_tsv_num[0], pitch*local_tsv_num[1], height)
    interval_x = lx/nx; interval_y = ly/ny; interval_z = lz/nz
    x_ = np.arange(0.5, 0.5+nx)*interval_x-lx/2; y_ = np.arange(0.5, 0.5+ny)*interval_y-ly/2; z_ = np.arange(0.5, 0.5+nz)*interval_z-lz/2
    xx, yy, zz = np.meshgrid(x_, y_, z_)
    local_grid_points = np.vstack((xx.reshape((1, -1)), yy.reshape((1, -1)), zz.reshape((1, -1))))

    domain, ct, _ = gmshio.read_from_msh(str(output_dir/'mesh0.msh'), MPI.COMM_SELF, 0, gdim=3)
    sys.stdout.reconfigure(line_buffering=True)
    V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    uh = fem.Function(V)
    lambda_, mu, alpha = utils.assign_materials(domain, list(range(1, len(young_modulus)+1)), [lambda__, mu_, alpha_], ct)
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
        domain, ct, _ = gmshio.read_from_msh(str(output_dir/'mesh1.msh'), MPI.COMM_SELF, 0, gdim=3)
        sys.stdout.reconfigure(line_buffering=True)
        V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
        uh_dummy = fem.Function(V)
        lambda_dummy, mu_dummy, alpha_dummy = utils.assign_materials(domain, list(range(1, len(young_modulus)+1)), [lambda__, mu_, alpha_], ct)
        temperature_dummy = fem.Constant(domain, thermal_load)

        V_von_mises_dummy = fem.functionspace(domain, ("DG", 2))
        stresses_dummy = fem.Function(V_von_mises_dummy)

        bb_tree = geometry.bb_tree(domain, domain.geometry.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, local_grid_points.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, local_grid_points.T)
        local_cells_dummy = [colliding_cells.links(i)[0] for i,_ in enumerate(local_grid_points.T)]
        

    global_element_dofs = [(i, j, k, d) 
                           for i in range(interp_num[0]) 
                           for j in range(interp_num[1]) 
                           for k in range(interp_num[2]) 
                           for d in range(3)
                           if i*j*k==0 or i==interp_num[0]-1 or j==interp_num[1]-1 or k==interp_num[2]-1]
    local2global, _= ga.create_local2global_mapping(global_element_dofs, block_num[0]+2*dummy_block_num[0], block_num[1]+2*dummy_block_num[1], interp_num[0], interp_num[1])
    dummy_tags = ga.locate_dummy_blocks(block_num[0], block_num[1], dummy_block_num[0], dummy_block_num[1])

    rank_blocks = utils.get_rank_parts(np.arange((block_num[0]+2*dummy_block_num[0])*(block_num[1]+2*dummy_block_num[1])), rank, size)
    rank_values = np.empty((len(rank_blocks), local_grid_points.shape[1]), dtype=np.float64)
    for i, block in enumerate(rank_blocks):
        if dummy_tags[block] == 0:
            uh.x.array[:] = x[local2global[block]]@modes[:-1]+thermal_load*modes[-1]
            stress_expr = fem.Expression(von_mises(uh, lambda_, mu, alpha, temperature), V_von_mises.element.interpolation_points())
            stresses.interpolate(stress_expr)
            rank_values[i] = np.linalg.norm(stresses.eval(local_grid_points.T, local_cells), axis=1, keepdims=False)
        else:
            uh_dummy.x.array[:] = x[local2global[block]]@modes_dummy[:-1]+thermal_load*modes_dummy[-1]
            stress_expr = fem.Expression(von_mises(uh_dummy, lambda_dummy, mu_dummy, alpha_dummy, temperature_dummy), V_von_mises_dummy.element.interpolation_points())
            stresses_dummy.interpolate(stress_expr)
            rank_values[i] = np.linalg.norm(stresses_dummy.eval(local_grid_points.T, local_cells_dummy), axis=1, keepdims=False)
    
    values = None
    recv_counts = None
    displaments = None
    rank_blocks = comm.gather(len(rank_blocks), root=0)
    if rank == 0:
        values = np.empty(((block_num[0]+2*dummy_block_num[0])*(block_num[1]+2*dummy_block_num[1]), local_grid_points.shape[1]), dtype=np.float64).flatten()
        recv_counts = np.array(rank_blocks, dtype=np.int32)*local_grid_points.shape[1]
        displaments = np.insert(np.cumsum(recv_counts[:-1]), 0, 0)
    comm.Gatherv(rank_values.flatten(), [values, recv_counts, displaments, MPI.DOUBLE], root=0)

    if rank == 0:
        values = values.reshape((-1, local_grid_points.shape[1]))
        contour = utils.reshape_to_contour(values, block_num[0]+2*dummy_block_num[0], block_num[1]+2*dummy_block_num[1], grid_num[0], grid_num[1], grid_num[2])
        np.save(output_dir/rst_fname, contour)

