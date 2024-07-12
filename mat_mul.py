import numpy as np
from cubyc import Run

@Run(tags=['linear_algebra'], branch='logan-test')
def matrix_multiplication(n_size: int):
    A = np.random.rand(n_size, n_size)
    B = np.random.rand(n_size, n_size)

    _ = np.dot(A, B)

for n_size in [5, 7, 9]:
    matrix_multiplication(n_size=n_size)
