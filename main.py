import yaml
import os
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

    os.system('python mesh.py --config {}'
              .format(args.config))

    os.system('mpirun -np {} python local.py --config {}'
                .format(config['parallel']['local']['mpi'],
                        args.config))

    for i, array in enumerate(config['tsv_array']):
        os.system('mpirun -np {} python global.py --config {} --array {}'
                  .format(config['parallel']['global']['mpi'],
                          args.config,
                          i))
        
        os.system('mpirun -np {} python post.py --config {} --array {}'
                  .format(config['parallel']['post']['mpi'],
                          args.config,
                          i))
