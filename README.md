# MORE-Stress: Model Order Reduction based Efficient Numerical Algorithm for Thermal Stress Simulation of TSV Arrays in 2.5D/3D IC
## Overview
This repo contains the implementation of **MORE-Stress**, a novel numerical algorithm
for efficient thermal stress simulation of TSV arrays in 2.5D/3D ICs based on finite element method and model
order reduction. The detailed methodology is described in the paper.
## Requirements
```
conda env create -f env.yml 
```
Our algorithm is built upon the **FeniCS** framework. For more information, please refer to the project's [homepage](https://fenicsproject.org/).
## Usage
To perform the whole process, execute:
```
python main.py --config config.yml
```
To perform each stage seperately, execute:
```
python mesh.py --config config.yml (local mesh)
mpirun -np xxx python local.py --config config.yml (one-shot local stage)
mpirun -np xxx python global.py --config config.yml --array xxx (global stage for array xxx)
mpirun -np xxx python post.py --config config.yml --array xxx (post-processing for array xxx)
```
## License
This repo is released under the MIT License.
## Citation
If you think our work is useful, please feel free to cite our paper. (coming soon)
## Contact
For any questions, please do not hesitate to contact us.
```
txzhu@pku.edu.cn
```
