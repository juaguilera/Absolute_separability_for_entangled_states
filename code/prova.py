import numpy as np
import math
from toqito.channels import partial_transpose
from toqito.perms import symmetric_projection

# Parameters
d = 2  # Local dimension of each qudit
N = 2  # Number of qudits

# Symmetric projection operator
IS = symmetric_projection(d, N)

# Dimension of the symmetric subspace
dim_symmetric = math.comb(d + N - 1, N)

# Normalize the symmetric projection operator
IS_normalized = IS / dim_symmetric

# Largest bipartition size
biggest_bipartition = int(math.floor(N / 2))

# Dimensions for the bipartition
dim = [d**biggest_bipartition, d**(N - biggest_bipartition)]

# Partial transpose of the normalized symmetric projection
IS_T = partial_transpose(IS_normalized, 0, dim)

# Compute eigenvalues of the partial transpose
eigenvalues = np.linalg.eigvalsh(IS_T)

# Sort eigenvalues to find the minimal one
sorted_eigenvalues = np.sort(eigenvalues)
lambda_min = sorted_eigenvalues[0]  # Minimal eigenvalue after normalization

# Additional scaling for symmetric subspace and normalization
scaling_factor = math.comb(N, biggest_bipartition)

# Compute alpha_minus based on the normalized eigenvalue
alpha_minus_theoretical = -lambda_min * scaling_factor

# Output results
print("Lambda_min (minimal eigenvalue after normalization):", lambda_min)
print("Scaling factor (binomial coefficient):", scaling_factor)
print("Alpha -:", scaling_factor*lambda_min)
print("Alpha_- (theoretical with normalization and scaling):", alpha_minus_theoretical)

# Verification
expected_alpha_minus = -0.75
if np.isclose(alpha_minus_theoretical, expected_alpha_minus, atol=1e-3):
    print("Alpha_- matches the expected value of -0.75.")
else:
    print("Alpha_- does not match the expected value. Please check assumptions.")
