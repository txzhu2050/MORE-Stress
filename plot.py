import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

output_dir = Path(config['output_dir'])
if not output_dir.exists():
    raise ValueError("Invalid path")

data = np.load(output_dir/'vonmises2.npy')

plt.imshow(data[0, :, :])
plt.savefig(output_dir/'vonmises2.png')

# self-defined plots ...