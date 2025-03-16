import yaml
import os
import sys
import gmsh
import numpy as np
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config_file")

def build_geometry_and_generate_mesh(material_tags, pitch=None, diameter=None, thickness=None, height=None, local_tsv_num=None, lcar1=None, lcar2=None, fname='mesh.msh', isdummy=False):
    gmsh.initialize()
    tag = gmsh.model.occ.addBox(-pitch/2*local_tsv_num["x"], -pitch/2*local_tsv_num["y"], -height/2, pitch*local_tsv_num["x"], pitch*local_tsv_num["y"], height)
    if not isdummy:
        blocks = [(3, tag)]
        tsvs = []
        liners = []
        for i in range(local_tsv_num["x"]):
            for j in range(local_tsv_num["y"]):
                tag = gmsh.model.occ.addCylinder(-pitch/2*local_tsv_num["x"] + pitch*(i+1/2), -pitch/2*local_tsv_num["y"] + pitch*(j+1/2), -height/2, 0, 0, height, diameter/2)
                tsvs.append((3, tag))
                tag = gmsh.model.occ.addCylinder(-pitch/2*local_tsv_num["x"] + pitch*(i+1/2), -pitch/2*local_tsv_num["y"] + pitch*(j+1/2), -height/2, 0, 0, height, diameter/2+thickness)
                liners.append((3, tag))
        gmsh.model.occ.fragment(blocks, liners)
        gmsh.model.occ.fragment(liners, tsvs)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        tsvs = []
        liners = []
        blocks = []
        for volume in volumes:
            mass = gmsh.model.occ.getMass(volume[0], volume[1])
            if np.isclose(mass, height*(diameter/2)**2*np.pi):
                tsvs.append(volume)
            elif np.isclose(mass, height*((diameter/2+thickness)**2-(diameter/2)**2)*np.pi):
                liners.append(volume)
            else:
                blocks.append(volume)
        gmsh.model.addPhysicalGroup(3, [block[1] for block in blocks], tag=material_tags["Si"])
        gmsh.model.addPhysicalGroup(3, [tsv[1] for tsv in tsvs], tag=material_tags["Cu"])
        gmsh.model.addPhysicalGroup(3, [liner[1] for liner in liners], tag=material_tags["SiO2"])

        surfaces = gmsh.model.getEntities(2)
        block_surface = []
        for surface in surfaces:
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if np.allclose(com, [0, 0, height/2]) or np.allclose(com, [0, 0, -height/2]):
                block_surface.append(surface[1])
        gmsh.model.addPhysicalGroup(2, block_surface)

        gmsh.model.mesh.setSize(gmsh.model.getBoundary(blocks, False, False, True), lcar1)
        #gmsh.model.mesh.setSize(gmsh.model.getBoundary(tsvs, False, False, True), lcar2)
        gmsh.model.mesh.setSize(gmsh.model.getBoundary(liners, False, False, True), lcar2)
        gmsh.model.mesh.generate(3)
    else:
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [tag], tag=material_tags["Si"])
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lcar1)
        gmsh.model.mesh.generate(3)
    gmsh.write(str(fname))
    gmsh.finalize()

    return 

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    output_dir = Path(config['output_dir'])
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    materials = config['material']
    material_tags = {}
    E = ''
    v = ''
    alpha = ''
    for i, material in enumerate(materials):
        material_tags[material['name']] = i+1
        E = E + str(materials[i]['young_modulus']) + ' '
        v = v + str(materials[i]['poisson_ratio']) + ' '
        alpha = alpha + str(materials[i]['thermal_expansion_coefficient']) + ' '
    E = E[:-1]; v = v[:-1]; alpha = alpha[:-1]

    thermal_load = config['temperature']

    thickness = config['geometry']['thickness']
    pitch = config['geometry']['pitch']
    diameter = config['geometry']['diameter']
    height = config['geometry']['height']

    solver_config = config['solver']
    block_tsv_num = solver_config['block_tsv_num']

    interp_num = solver_config['interp_num']
    dummy_block_num = solver_config['dummy_block_num']
    iter_options = solver_config['iterative_solver']

    grid_num = config['plot']

    stage = config['stage']

    msh_fname = ['mesh0.msh', 'mesh1.msh']
    if stage['mesh']:
        build_geometry_and_generate_mesh(material_tags, pitch, diameter, thickness, height, block_tsv_num, lcar1=pitch/10, lcar2=diameter/10, fname=output_dir/msh_fname[0], isdummy=False)
        if dummy_block_num['x'] > 0 or dummy_block_num['y'] > 0:
            build_geometry_and_generate_mesh(material_tags, pitch=pitch, height=height, local_tsv_num=block_tsv_num, lcar1=pitch/10, fname=output_dir/msh_fname[1], isdummy=True)
        sys.stdout.reconfigure(line_buffering=True)
    if stage['local']:
        print("\nStart calculating local modes!")
        print("\nFilled:")
        os.system('mpirun -np {} python local.py -n {} -tag {} -dir {} -E {} -v {} -alpha {} -T {} -p {} -d {} -ht {} -nt {} {} -ni {} {} {}'
                    .format(config['parallel']['local']['mpi'],
                            config['parallel']['local']['omp'],
                            0,
                            str(output_dir),
                            E, v, alpha, thermal_load,
                            pitch, diameter, height,
                            block_tsv_num["x"], block_tsv_num["y"],
                            interp_num['x'], interp_num['y'], interp_num['z']))
        if dummy_block_num['x'] > 0 or dummy_block_num['y'] > 0:
            print("\nDummy:")
            os.system('mpirun -np {} python local.py -n {} -tag {} -dir {} -E {} -v {} -alpha {} -T {} -p {} -d {} -ht {} -nt {} {} -ni {} {} {}'
                        .format(config['parallel']['local']['mpi'],
                                config['parallel']['local']['omp'],
                                1,
                                str(output_dir),
                                E, v, alpha, thermal_load,
                                pitch, diameter, height,
                                block_tsv_num["x"], block_tsv_num["y"],
                                interp_num['x'], interp_num['y'], interp_num['z']))
    for i, array in enumerate(config['tsv_array']):
        tsv_num = array['tsv_num']
        block_num_x = tsv_num['x']//block_tsv_num['x']
        block_num_y = tsv_num['y']//block_tsv_num['y']
        sol_fname = 'x'+str(i)
        rst_fname = 'vonmises'+str(i) #this can be changed to other fields
        if stage['global']:
            print("\nStart calculating global problems for tsv array {}!\n".format(i))
            os.system('mpirun -np {} python global.py -sol_fname {} -dir {} -ni {} {} {} -nb {} {} -nd {} {} -atol {} -rtol {} -max_it {}'
                    .format(config['parallel']['global']['mpi'],
                            sol_fname, str(output_dir),
                            interp_num['x'], interp_num['y'], interp_num['z'],
                            block_num_x, block_num_y,
                            dummy_block_num['x'], dummy_block_num['y'],
                            iter_options['atol'], iter_options['rtol'], iter_options['max_it']))
        if stage['post_processing']:
            print("\nStart post-processing for tsv array {}!\n".format(i))
            os.system('mpirun -np {} python post_processing.py -sol_fname {} -rst_fname {} -dir {} -E {} -v {} -alpha {} -T {} -p {} -ht {} -nt {} {} -ni {} {} {} -nb {} {} -nd {} {} -ng {} {} {}'
                    .format(config['parallel']['post_processing']['mpi'],
                            sol_fname, rst_fname, str(output_dir),
                            E, v, alpha, thermal_load,
                            pitch, height,
                            block_tsv_num["x"], block_tsv_num["y"],
                            interp_num['x'], interp_num['y'], interp_num['z'],
                            block_num_x, block_num_y,
                            dummy_block_num['x'], dummy_block_num['y'],
                            grid_num['x'], grid_num['y'], grid_num['z']))
