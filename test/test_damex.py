# %% 
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp
import MLExtreme as mlx
# import pdb

radius = 3
epsilon = 0.3
X_test = np.array([
    [12, 13, 14],
    [1, 1, 1],
    [12, 0.5, 14],
    [12, 5, 14],
    [1, 2, 3],
    [12, 0.5, 0.5],
    [12, 0.5, 0.5],
    [0.5, 0.5, 12]])

bin_array = mlx.binary_large_features(X_test, radius, epsilon=epsilon)
bin_array
faces, counts = mlx.damex_0(bin_array)

faces2, limit_mass = mlx.damex(X_test, radius, epsilon=epsilon, min_counts=0,
                               standardize=False)


faces2, limit_mass = mlx.damex(X_test, radius, epsilon=epsilon, min_counts=0,
                               standardize=True)

print(faces)
print(faces2)
print(counts)
print(limit_mass)

Xt = mlx.rank_transform(X_test)

clef_faces = mlx.clef(Xt, 3, 0.1)

clef_faces
faces
