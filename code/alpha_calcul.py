import numpy as np
from scipy.linalg import expm, eigh
from scipy.optimize import minimize

def generate_random_unitary(dim):
    """Generates a random unitary matrix of dimension dim."""
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q

def partial_transpose(rho, dims, transpose_on):
    """
    Perform the partial transpose of a density matrix `rho`.

    Parameters:
    - rho: The density matrix.
    - dims: List of subsystem dimensions.
    - transpose_on: Index of the subsystem to transpose.

    Returns:
    - Partial transpose of `rho`.
    """
    #dim = np.prod(dims)
    reshaped_rho = rho.reshape(*dims, *dims)
    transposed_rho = np.swapaxes(
        reshaped_rho,
        axis1=transpose_on,
        axis2=transpose_on + len(dims)
    )
    return transposed_rho.reshape(rho.shape) # (dim, dim)

def eigenvalue_minimization(rho, dims, N):
    """
    Minimizes the smallest eigenvalue of the partial transpose of the density matrix.

    Parameters:
    - rho: Initial density matrix.
    - dims: List of subsystem dimensions.

    Returns:
    - The minimum eigenvalue.
    """
    dim = int(np.sqrt(rho.size))  # Calculate dim based on the size of rho
    if dim * dim != rho.size:
        raise ValueError(f"rho must have a size that is a perfect square, but has size {rho.size}")

    dim = dims[0]
    def objective(unitary_params):
        """Objective function to minimize."""

        # Ensure the size of unitary_params is correct
        if unitary_params.size != dim * dim:
            raise ValueError(f"unitary_params must have size {dim * dim}, but has size {unitary_params.size}")

        # Reshape the unitary parameters back into a matrix
        unitary_matrix = unitary_params.reshape(dim, dim)
        # Build a unitary matrix from the parameters
        U, _ = np.linalg.qr(unitary_matrix)

        # Reshape rho to dim x dim for proper matrix multiplication
        #reshaped_rho = rho.reshape(dims)
        U_full = U

        for i in range(N):
            U_full = np.kron(U_full, U)

        rotated_rho = U_full @ rho @ U_full.conj().T
        pt_rho = partial_transpose(rotated_rho, dims, transpose_on=0)  # Transpose first subsystem
        eigenvalues = eigh(pt_rho, eigvals_only=True)
        return eigenvalues[0]  # Smallest eigenvalue

    # Initial guess: Identity transformation (no rotation), flattened into 1D
    #dim = dims[0]
    initial_params = np.zeros((dim, dim)).flatten()

    # Perform the optimization
    result = minimize(objective, initial_params, method="BFGS")
    return result.fun


# Parameters for the system
d = 2  # Local dimension of each subsystem
N = 2  # Number of subsystems
dims = [d] * N

# Generate a random initial density matrix
random_state = np.random.randn(d ** N) + 1j * np.random.randn(d ** N)
random_state /= np.linalg.norm(random_state)
rho = np.outer(random_state, random_state.conj())

# Approximate alpha_-
alpha_minus = eigenvalue_minimization(rho, dims, N)
print(f"Approximate alpha_-: {alpha_minus}")
