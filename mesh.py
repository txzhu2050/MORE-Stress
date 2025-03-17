import numpy as np
import gmsh
import yaml
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
args = parser.parse_args()

import sys

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
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    output_dir = Path(config['output_dir'])
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    materials = config['material']
    material_tags = {}
    for i, material in enumerate(materials):
        material_tags[material['name']] = i+1

    thickness = config['geometry']['thickness']
    pitch = config['geometry']['pitch']
    diameter = config['geometry']['diameter']
    height = config['geometry']['height']

    block_tsv_num = config['solver']['block_tsv_num']

    build_geometry_and_generate_mesh(material_tags, pitch, diameter, thickness, height, block_tsv_num, lcar1=pitch/10, lcar2=diameter/10, fname=output_dir/'mesh0.msh', isdummy=False)
    build_geometry_and_generate_mesh(material_tags, pitch=pitch, height=height, local_tsv_num=block_tsv_num, lcar1=pitch/10, fname=output_dir/'mesh1.msh', isdummy=True)
    sys.stdout.reconfigure(line_buffering=True)