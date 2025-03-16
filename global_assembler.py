from dolfinx import fem
from petsc4py import PETSc
import numpy as np

def calculate_element_stiffness(bilinear_form, modes):
    A = fem.assemble_matrix(bilinear_form).to_scipy()

    return modes@(A@modes.T)

def calculate_element_load(linear_form, modes):
    b = fem.assemble_vector(linear_form).array
    
    return modes@b

def create_local2global_mapping(element_dofs, block_num_x, block_num_y, x_num, y_num):
    local2global = np.empty((block_num_x*block_num_y, len(element_dofs)), dtype=np.int32)
    dof_num = len(element_dofs)
    dof_local = np.array([i for i in range(len(element_dofs))], dtype=np.int32)
    local2global[0] = dof_local

    node_tag = {"left_bottom":  dof_local[[True if i==0 and j==0 else False for i,j,_,_ in element_dofs]],
                "left_top":     dof_local[[True if i==0 and j==y_num-1 else False for i,j,_,_ in element_dofs]],
                "right_top":    dof_local[[True if i==x_num-1 and j==y_num-1 else False for i,j,_,_ in element_dofs]],
                "right_bottom": dof_local[[True if i==x_num-1 and j==0 else False for i,j,_,_ in element_dofs]],
                "left":         dof_local[[True if i==0 and j!=0 and j!=y_num-1 else False for i,j,_,_ in element_dofs]],
                "top":          dof_local[[True if j==y_num-1 and i!=0 and i!=x_num-1 else False for i,j,_,_ in element_dofs]],
                "right":        dof_local[[True if i==x_num-1 and j!=0 and j!=y_num-1 else False for i,j,_,_ in element_dofs]],
                "bottom":       dof_local[[True if j==0 and i!=0 and i!=x_num-1 else False for i,j,_,_ in element_dofs]],
                }

    dof_delta = len(np.setdiff1d(dof_local, np.concatenate([node_tag["left_bottom"], node_tag["left"], node_tag["left_top"]])))
    for i in range(1, block_num_x):
        local2global[i][np.concatenate([node_tag["left_bottom"], node_tag["left"], node_tag["left_top"]])] = local2global[i-1][np.concatenate([node_tag["right_bottom"], node_tag["right"], node_tag["right_top"]])]
        local2global[i][np.setdiff1d(dof_local, np.concatenate([node_tag["left_bottom"], node_tag["left"], node_tag["left_top"]]))] = np.array([dof_num+i for i in range(dof_delta)], dtype=np.int32)
        dof_num += dof_delta

    for i in range(1, block_num_y):
        dof_delta = len(np.setdiff1d(dof_local, np.concatenate([node_tag["left_bottom"], node_tag["bottom"], node_tag["right_bottom"]])))
        local2global[block_num_x*i][np.concatenate([node_tag["left_bottom"], node_tag["bottom"], node_tag["right_bottom"]])] = local2global[block_num_x*(i-1)][np.concatenate([node_tag["left_top"], node_tag["top"], node_tag["right_top"]])]
        local2global[block_num_x*i][np.setdiff1d(dof_local, np.concatenate([node_tag["left_bottom"], node_tag["bottom"], node_tag["right_bottom"]]))] = np.array([dof_num+i for i in range(dof_delta)], dtype=np.int32)
        dof_num += dof_delta

        dof_delta = len(np.setdiff1d(dof_local, np.concatenate([node_tag["left_bottom"], node_tag["bottom"], node_tag["right_bottom"], node_tag["left"], node_tag["left_top"]])))
        for j in range(1, block_num_x):
            local2global[block_num_x*i+j][np.concatenate([node_tag["left"], node_tag["left_top"]])] = local2global[block_num_x*i+j-1][np.concatenate([node_tag["right"], node_tag["right_top"]])]
            local2global[block_num_x*i+j][np.concatenate([node_tag["bottom"], node_tag["right_bottom"]])] = local2global[block_num_x*(i-1)+j][np.concatenate([node_tag["top"], node_tag["right_top"]])]
            local2global[block_num_x*i+j][node_tag["left_bottom"]] = local2global[block_num_x*(i-1)+j-1][node_tag["right_top"]]
            local2global[block_num_x*i+j][np.setdiff1d(dof_local, np.concatenate([node_tag["left_bottom"], node_tag["bottom"], node_tag["right_bottom"], node_tag["left"], node_tag["left_top"]]))] = np.array([dof_num+i for i in range(dof_delta)], dtype=np.int32)
            dof_num += dof_delta

    return local2global, dof_num

def create_3D_view(local2global, element_dofs, block_num_x, block_num_y, x_num, y_num, z_num):
    view = np.zeros((z_num, block_num_y*(y_num-1)+1, block_num_x*(x_num-1)+1), dtype=np.int32)
    for block in range(block_num_x*block_num_y):
        idx_y = block // block_num_x
        idx_x = block % block_num_x
        for i,dof in enumerate(element_dofs):
            view[dof[2]][idx_y*(y_num-1)+dof[1]][idx_x*(x_num-1)+dof[0]] = local2global[block][i]

    return view

def locate_dummy_blocks(block_num_x, block_num_y, dummy_block_num_x, dummy_block_num_y):
    dummy_tags = np.zeros((block_num_x+2*dummy_block_num_x)*(block_num_y+2*dummy_block_num_y), dtype=np.int32)
    for i in range(block_num_y+2*dummy_block_num_y):
        for j in range(block_num_x+2*dummy_block_num_x):
            if i < dummy_block_num_y or i > block_num_y+dummy_block_num_y-1 or j < dummy_block_num_x or j > block_num_x+dummy_block_num_x-1:
                dummy_tags[i*(block_num_x+2*dummy_block_num_x)+j] = 1
    
    return dummy_tags

def assemble_stiffness(A_petsc, local2global, element_stiffness, dummy_tags, rank, size):
    cells_per_rank = len(local2global)//size
    res = len(local2global)%size
    if rank<res:
        blocks = range((cells_per_rank+1)*rank, (cells_per_rank+1)*(rank+1))
    else:
        blocks = range((cells_per_rank+1)*res+cells_per_rank*(rank-res), (cells_per_rank+1)*res+cells_per_rank*(rank+1-res))
    for block in blocks:
        global_dofs = local2global[block]
        stiffness = element_stiffness[dummy_tags[block]]
        A_petsc.setValuesBlocked(global_dofs, global_dofs, stiffness, addv=PETSc.InsertMode.ADD_VALUES)

    return

def assemble_load(b_petsc, local2global, element_load, dummy_tags, rank, size):
    cells_per_rank = len(local2global)//size
    res = len(local2global)%size
    if rank<res:
        t = range((cells_per_rank+1)*rank, (cells_per_rank+1)*(rank+1))
    else:
        t = range((cells_per_rank+1)*res+cells_per_rank*(rank-res), (cells_per_rank+1)*res+cells_per_rank*(rank+1-res))
    for block in t:
        global_dofs = local2global[block]
        load = element_load[dummy_tags[block]]
        b_petsc.setValuesBlocked(global_dofs, load, addv=PETSc.InsertMode.ADD_VALUES)

    return

def set_bc(b_petsc, dofs, bc_values):
    for i, bc_value in enumerate(bc_values):
        b_petsc.setValues(dofs[i], bc_value, addv=PETSc.InsertMode.INSERT_VALUES)

    return

if __name__ == "__main__":
    pass