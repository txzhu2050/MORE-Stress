import yaml
import os
import numpy as np
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    output_dir = Path(config['output_dir'])
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    dummy_block_num = config['solver']['dummy_block_num']

    os.system('python mesh.py --config {}'
              .format(args.config))
    print("\nStart calculating local modes!")
    print("\nFilled:")
    os.system('mpirun -np {} python local.py --nthreads {} --config {} --tag {}'
                .format(config['parallel']['local']['mpi'],
                        config['parallel']['local']['omp'],
                        args.config,
                        0))
    if dummy_block_num['x'] > 0 or dummy_block_num['y'] > 0:
        print("\nDummy:")
        os.system('mpirun -np {} python local.py --nthreads {} --config {} --tag {}'
                    .format(config['parallel']['local']['mpi'],
                            config['parallel']['local']['omp'],
                            args.config,
                            1))
    for i, array in enumerate(config['tsv_array']):
        print("\nStart calculating the global problem for tsv array {}!\n".format(i))
        os.system('mpirun -np {} python global.py --config {} --array {}'
                  .format(config['parallel']['global']['mpi'],
                          args.config,
                          i))
        print("\nStart post-processing for tsv array {}!\n".format(i))
        rst_fname = 'vonmises'+str(i) #this can be changed to other fields
        os.system('mpirun -np {} python post.py --config {} --array {} --rst_fname {}'
                  .format(config['parallel']['post']['mpi'],
                  args.config,
                  i,
                  rst_fname))
