from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
args = parser.parse_args()

import yaml

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

import os
os.environ["OMP_NUM_THREADS"] = str(config['parallel']['local']['omp'])

from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, apply_lifting, set_bc
from dolfinx.io import gmshio
import ufl
import numpy as np
from petsc4py import PETSc
from functools import partial
import global_assembler as ga
import utils
from pathlib import Path
import sys

if __name__ == "__main__":
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

    block_tsv_num = config['solver']['block_tsv_num']
    interp_num = config['solver']['interp_num']

    for tag in range(2):
        if rank == 0:
            print('\nStart calculating local modes!')
        comm.Barrier()
        
        if comm.rank == 0:
            domain, ct, _ = gmshio.read_from_msh(str(output_dir/('mesh'+str(tag)+'.msh')), MPI.COMM_SELF, 0, gdim=3)
        else:
            with utils.suppress_stdout_stderr():
                domain, ct, _ = gmshio.read_from_msh(str(output_dir/('mesh'+str(tag)+'.msh')), MPI.COMM_SELF, 0, gdim=3)
        sys.stdout.reconfigure(line_buffering=True)

        lambda_, mu, alpha = utils.assign_materials(domain, list(range(1, len(E)+1)), [lambda__, mu_, alpha_], ct)
        temperature = fem.Constant(domain, thermal_load)

        tdim = domain.topology.dim
        V = fem.functionspace(domain, ("Lagrange", 2, (tdim,)))
        local_dof_num = V.dofmap.index_map.size_local*V.dofmap.index_map_bs
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = partial(utils.a, lambda_=lambda_, mu=mu)
        L = partial(utils.L, alpha=alpha, temperature=temperature)
        bilinear_form = fem.form(a(u, v))
        linear_form = fem.form(L(v))

        domain.topology.create_connectivity(tdim - 1, tdim)
        local_boundary_facets = mesh.exterior_facet_indices(domain.topology)
        local_boundary_dofs = fem.locate_dofs_topological(V, tdim-1, local_boundary_facets)

        pseudo_bc = fem.dirichletbc(fem.Constant(domain, default_scalar_type((0, 0, 0))), local_boundary_dofs, V)
        A_petsc = assemble_matrix(bilinear_form, [pseudo_bc])
        A_petsc.assemble()
        b_petsc = create_vector(linear_form)

        solver = PETSc.KSP().create(MPI.COMM_SELF)
        solver.setOperators(A_petsc)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        solver.getPC().setFactorSolverType('mumps')

        print(f'Decomposition is being performed on rank {rank} ...')
        solver.setUp() #time-consuming decomposition of A

        global_element_dofs = [(i, j, k, d) 
                            for i in range(interp_num['x']) 
                            for j in range(interp_num['y']) 
                            for k in range(interp_num['z']) 
                            for d in range(tdim)
                            if i*j*k==0 or i==interp_num['x']-1 or j==interp_num['y']-1 or k==interp_num['z']-1]
        
        rank_dofs = utils.get_rank_parts(global_element_dofs, rank, size)

        local_modes = np.empty((len(rank_dofs), local_dof_num), dtype=np.float64)
        if rank == size-1:
            local_modes = np.empty((len(rank_dofs)+1, local_dof_num), dtype=np.float64)

        temperature.value = 0.0
        uh = fem.Function(V)
        u_bc = fem.Function(V)
        lagrange_interp = partial(utils.lagrange_interpolation,
                                interp_num=(interp_num['x'], interp_num['y'], interp_num['z']),
                                scale=(pitch*block_tsv_num['x'], pitch*block_tsv_num['y'], height))
        for i, dof in enumerate(rank_dofs):
            u_D = partial(lagrange_interp, interp_point=dof)
            u_bc.interpolate(u_D)
            bc = fem.dirichletbc(u_bc, local_boundary_dofs)
            with b_petsc.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b_petsc, linear_form)
            apply_lifting(b_petsc, [bilinear_form], [[bc]])
            b_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b_petsc, [bc])
            b_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
            solver.solve(b_petsc, uh.vector)
            uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
            local_modes[i] = uh.x.array
            print(f"{dof} mode calculated on rank {rank}!")

        if rank == size-1:
            temperature.value = 1.0
            with b_petsc.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b_petsc, linear_form)
            apply_lifting(b_petsc, [bilinear_form], [[pseudo_bc]])
            b_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b_petsc, [pseudo_bc])
            b_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
            solver.solve(b_petsc, uh.vector)
            uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
            local_modes[-1] = uh.x.array
            print(f"T mode calculated on rank {rank}!")

        comm.Barrier()

        modes = None
        recv_counts = None
        displaments = None
        rank_dofs = comm.gather(len(local_modes), root=0)
        if rank == 0:
            modes = np.empty((len(global_element_dofs)+1, local_dof_num), dtype=np.float64).flatten()
            recv_counts = np.array(rank_dofs, dtype=np.int32)*local_dof_num
            displaments = np.insert(np.cumsum(recv_counts[:-1]), 0, 0)
        comm.Gatherv(local_modes.flatten(), [modes, recv_counts, displaments, MPI.DOUBLE], root=0)

        if rank == 0:
            temperature.value = thermal_load
            modes = modes.reshape((-1, local_dof_num))
            global_element_stiffness = ga.calculate_element_stiffness(bilinear_form, modes[:-1])
            global_element_load = ga.calculate_element_load(linear_form, modes[:-1])
            np.save(output_dir/('stiffness'+str(tag)), global_element_stiffness)
            np.save(output_dir/('load'+str(tag)), global_element_load)
            np.save(output_dir/('modes'+str(tag)), modes)
        
        comm.Barrier()
