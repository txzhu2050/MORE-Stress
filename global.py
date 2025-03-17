import os
os.environ["OMP_NUM_THREADS"] = "1"

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--array', type=int)
args = parser.parse_args()

from petsc4py import PETSc
import numpy as np
import global_assembler as ga
import yaml
from pathlib import Path
import sys
sys.stdout.reconfigure(line_buffering=True)

if __name__ == "__main__":
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    output_dir = Path(config['output_dir'])
    if not output_dir.exists():
        raise ValueError("Invalid path")

    dummy_block_num = config['solver']['dummy_block_num']
    interp_num = config['solver']['interp_num']
    block_tsv_num = config['solver']['block_tsv_num']

    tsv_num = config['tsv_array'][args.array]['tsv_num']
    block_num = {'x':tsv_num['x']//block_tsv_num['x'], 'y':tsv_num['y']//block_tsv_num['y']}

    dummy_tag = 0
    if dummy_block_num['x'] > 0 or dummy_block_num['y'] > 0:
        dummy_tag = 1
    
    global_element_dofs = [(i, j, k, d) 
                           for i in range(interp_num['x']) 
                           for j in range(interp_num['y']) 
                           for k in range(interp_num['z']) 
                           for d in range(1)
                           if i*j*k==0 or i==interp_num['x']-1 or j==interp_num['y']-1 or k==interp_num['z']-1]
    global_element_stiffness = np.load(output_dir/'stiffness0.npy')
    global_element_load = np.load(output_dir/'load0.npy')
    global_element_stiffness_dummy = None
    global_element_load_dummy = None
    if dummy_tag:
        global_element_stiffness_dummy = np.load(output_dir/'stiffness1.npy')
        global_element_load_dummy = np.load(output_dir/'load1.npy')

    local2global, global_dof_num = ga.create_local2global_mapping(global_element_dofs, block_num['x']+2*dummy_block_num['x'], block_num['y']+2*dummy_block_num['y'], interp_num['x'], interp_num['y'])
    dummy_tags = ga.locate_dummy_blocks(block_num['x'], block_num['y'], dummy_block_num['x'], dummy_block_num['y'])
    A_petsc = PETSc.Mat().createBAIJ(size=global_dof_num*3, bsize=3, comm=comm)
    A_petsc.setUp()
    b_petsc = PETSc.Vec().createMPI(size=global_dof_num*3, bsize=3, comm=comm)
    b_petsc.setUp()
    x_petsc = b_petsc.duplicate()

    ga.assemble_stiffness(A_petsc, local2global, [global_element_stiffness, global_element_stiffness_dummy], dummy_tags, rank, size)
    A_petsc.assemblyBegin()
    A_petsc.assemblyEnd()
    ga.assemble_load(b_petsc, local2global, [global_element_load, global_element_load_dummy], dummy_tags, rank, size)
    b_petsc.assemblyBegin()
    b_petsc.assemblyEnd()
    
    #some examples of assigning boundary conditions

    #1. clamp the top and bottom
    view = ga.create_3D_view(local2global, global_element_dofs, block_num['x']+2*dummy_block_num['x'], block_num['y']+2*dummy_block_num['y'], interp_num['x'], interp_num['y'], interp_num['z'])
    bottom_nodes = view[0,:,:].flatten()
    bottom_dofs = np.concatenate((bottom_nodes*3, bottom_nodes*3+1, bottom_nodes*3+2), dtype=np.int32)
    top_nodes = view[interp_num['z']-1,:,:].flatten()
    top_dofs = np.concatenate((top_nodes*3, top_nodes*3+1, top_nodes*3+2), dtype=np.int32)
    A_petsc.zeroRows(np.concatenate((bottom_dofs, top_dofs)), diag=1.0)
    A_petsc.assemblyBegin()
    A_petsc.assemblyEnd()
    ga.set_bc(b_petsc, [bottom_dofs, top_dofs], [np.full_like(bottom_dofs, 0.), np.full_like(top_dofs, 0.)])
    b_petsc.assemblyBegin()
    b_petsc.assemblyEnd()
    

    '''
    #2. supress rigid motion
    view = ga.create_3D_view(local2global, global_element_dofs, block_num['x']+2*dummy_block_num['x'], block_num['y']+2*dummy_block_num['y'], interp_num['x'], interp_num['y'], interp_num['z'])
    corner_nodes = [view[0,0,0], view[0,0,(block_num['x']+2*dummy_block_num['x'])*(interp_num['x']-1)], view[0,(block_num['y']+2*dummy_block_num['y'])*(interp_num['y']-1),0]]
    corner_dofs = np.array([corner_nodes[0], corner_nodes[0]+1, corner_nodes[0]+2, corner_nodes[1]+1, corner_nodes[1]+2, corner_nodes[2]+2], dtype=np.int32)
    A_petsc.zeroRows(corner_dofs, diag=1.0)
    A_petsc.assemblyBegin()
    A_petsc.assemblyEnd()
    ga.set_bc(b_petsc, [corner_dofs], [np.full_like(corner_dofs, 0.)])
    b_petsc.assemblyBegin()
    b_petsc.assemblyEnd()
    '''

    '''
    #3. set arbitary boundary conditions
    view = ga.create_3D_view(local2global, global_element_dofs, block_num['x']+2*dummy_block_num['x'], block_num['y']+2*dummy_block_num['y'], interp_num['x'], interp_num['y'], interp_num['z'])
    faces = [view[:,:,0].flatten(),
             view[:,:,(block_num['x']+2*dummy_block_num['x'])*(interp_num['x']-1)].flatten(),
             view[:,0,:].flatten(),
             view[:,(block_num['y']+2*dummy_block_num['y'])*(interp_num['y']-1),:].flatten(),
             view[interp_num['z']-1,:,:].flatten(),
             view[0,:,:].flatten()]
    boundary_nodes = np.unique(np.concatenate(faces))
    boundary_dofs = np.concatenate((boundary_nodes, boundary_nodes+1, boundary_nodes+2), dtype=np.int32)
    local_prescribed_nodes = np.array([], dtype=np.int32)
    for i in range(6):
        if rank == i:
            df = pd.read_csv('./face'+str(i)+'.csv', skiprows=8) #change boundary displacement filenames
            disp = df.iloc[:, 2:].to_numpy(dtype=np.float64)
            face = faces[i]
            local_prescribed_nodes = face[~np.isnan(disp[:,0])]
            disp = disp[~np.isnan(disp[:,0]), :]
            ga.set_bc(b_petsc, [local_prescribed_nodes, local_prescribed_nodes+1, local_prescribed_nodes+2], [disp[:,0], disp[:,1], disp[:,2]])
    comm.Barrier()
    b_petsc.assemblyBegin()
    b_petsc.assemblyEnd()

    recv_counts = np.array(comm.allgather(len(local_prescribed_nodes)))
    displs = np.insert(np.cumsum(recv_counts[:-1]), 0, 0)
    prescribed_nodes = np.empty(np.sum(recv_counts),dtype=np.int32)
    comm.Allgatherv(local_prescribed_nodes, [prescribed_nodes, recv_counts, displs, MPI.INT32_T])
    prescribed_dofs = np.concatenate((prescribed_nodes, prescribed_nodes+1, prescribed_nodes+2), dtype=np.int32)
    A_petsc.zeroRows(prescribed_dofs, diag=1.0)
    A_petsc.assemblyBegin()
    A_petsc.assemblyEnd()
    '''

    def monitor(ksp, its, rnorm):
        if comm.rank == 0:
            if its % 5 == 0:
                print(f"Iteration: {its}, Residual norm: {rnorm}")
    
    iter_options = config['solver']['iterative_solver']
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A_petsc)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.BJACOBI)
    solver.setTolerances(rtol=iter_options['rtol'], atol=iter_options['atol'], max_it=iter_options['max_it'])
    if monitor:
        solver.setMonitor(monitor)
    solver.solve(b_petsc, x_petsc)

    scatter, to_x = PETSc.Scatter().toZero(x_petsc)
    scatter.scatter(x_petsc, to_x, addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
    if rank == 0:
        np.save(output_dir/('x'+str(args.array)), to_x.getArray())
