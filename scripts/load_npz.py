import numpy as np

# Load the .npz file
data = np.load('act_bc_multi.npz')

# Access the arrays
for key in data.files:
    print(f"Array for {key}:")
    print(data[key])
    print("--------------------")