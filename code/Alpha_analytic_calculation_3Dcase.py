import sympy as sp
from sympy import I, Matrix, symbols, conjugate, sqrt
from sympy.matrices import eye
from sympy import symbols
from scipy.optimize import minimize


#Definim les variables de manera simbòlica
a, b, c = symbols('a b c', complex=True) #Igual també es pot provar amb valors reals
alpha = symbols('alpha', real=True)
constraint = sp.Eq(conjugate(a)*a + conjugate(b)*b + conjugate(c)*c, 1) #Forcem que estiguin normalitzats a,b,c. 

# Dickes (estaria més bé una funció per dimensions generals, però bueno)
D0 = Matrix([[1], [0], [0], [0]])  # |D0⟩ = |00⟩
D1 = Matrix([[0], [1/sqrt(2)], [1/sqrt(2)], [0]])  # |D1⟩ = (|01⟩ + |10⟩) / sqrt(2)
D2 = Matrix([[0], [0], [0], [1]])  # |D2⟩ = |11⟩

DS0 = Matrix([[1], [0], [0]])
DS1 = Matrix([[0], [1], [0]])
DS2 = Matrix([[0], [0], [1]])

# Define the pure state psi = a*|D0⟩ + b*|D1⟩ + c*|D2⟩
psi = a * DS0 + b * DS1 + c * DS2

# Symmetric identity operator for the symmetric subspace (3x3)
IdentitySymmetric = eye(3)

# Projection operator |psi⟩⟨psi| in the symmetric basis
KetBraPsi = psi * psi.H

DickeStates = [D0, D1, D2]

# Operator: I_Symmetric + alpha * |psi⟩⟨psi|
OperatorSym = IdentitySymmetric + alpha * KetBraPsi
print('Lambda(rho)_S')
sp.pprint(OperatorSym)

# Compute the eigenvalues of the operator
eigenvalues_sym = OperatorSym.eigenvals()

# Print the eigenvalues in the symmetric basis
print("Eigenvalues in symmetric basis:")
for eig in eigenvalues_sym:
    print(eig)

# --- Step 2: Transform to the computational basis ---

OperatorComp = Matrix.zeros(4, 4)
for i in range(3):
    for j in range(3):
        # Add the contribution of each entry multiplied by |Di⟩⟨Dj|
        OperatorComp += OperatorSym[i, j] * (DickeStates[i] * DickeStates[j].H)

# Simplify the operator in the computational basis
#OperatorComp = OperatorComp.applyfunc(lambda x: sp.simplify(x))

# Display the operator in the computational basis
print("Operator in computational basis:")
sp.pprint(OperatorComp)


# Compute the eigenvalues in the computational basis
eigenvalues_comp = OperatorComp.eigenvals()

# Print the eigenvalues in the computational basis
print("\nEigenvalues in computational basis:")
for eig in eigenvalues_comp:
    print(eig)

# --- Step 3: Perform the partial transpose with respect to the second qubit ---

def partial_transpose(matrix):
    """Performs the partial transpose with respect to the second qubit for a 4x4 SymPy matrix."""
    # Ensure the input is a 4x4 matrix
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4.")
    
    # Initialize an empty 4x4 matrix for the result
    result = sp.Matrix.zeros(4, 4)
    
    # Define the indices for 2-qubit systems (each qubit has dimension 2)
    basis_indices = [(i, j) for i in range(4) for j in range(4)]
    
    # Map the 4x4 indices to 2x2x2x2 indices and perform the partial transpose
    for i, j in basis_indices:
        # Convert 1D indices to 2-qubit indices: i = (i1, i2), j = (j1, j2)
        i1, i2 = divmod(i, 2)
        j1, j2 = divmod(j, 2)
        
        # Partial transpose swaps the second qubit indices (i2 and j2)
        transposed_i = i1 * 2 + j2
        transposed_j = j1 * 2 + i2
        
        # Assign the value to the transposed position
        result[transposed_i, transposed_j] = matrix[i, j]
    
    return result


# Perform the partial transpose on the operator in the computational basis
OperatorPartialTranspose = partial_transpose(OperatorComp)
sp.pprint(OperatorPartialTranspose)

# Compute the eigenvalues of the partially transposed operator
eigenvalues_pt = OperatorPartialTranspose.eigenvals()

# Print the eigenvalues after partial transpose
print("\nEigenvalues after partial transpose:")
for eig in eigenvalues_pt:
    print(eig)

print(len(eigenvalues_pt))

#######################################################################################
# Convert the smallest eigenvalue expression into a function for numerical optimization
def smallest_eigenvalue_func(alpha_val, a_val, b_val, c_val):
    """Compute the smallest eigenvalue numerically for given alpha, a, b, c."""
    substitutions = {alpha: alpha_val, a: a_val, b: b_val, c: c_val}
    eval_eigenvalues = [
        sp.simplify(eig.subs(substitutions)).evalf()
        for eig in OperatorPartialTranspose.eigenvals()
    ]
    return min(eval_eigenvalues)

# Wrapper for scipy optimization
def objective(x):
    """Objective function for optimization."""
    alpha_val, a_val, b_val, c_val = x
    return smallest_eigenvalue_func(alpha_val, a_val, b_val, c_val)

# Constraint for normalization
def constraint(x):
    """Normalization constraint |a|^2 + |b|^2 + |c|^2 = 1."""
    _, a_val, b_val, c_val = x
    return abs(a_val)**2 + abs(b_val)**2 + abs(c_val)**2 - 1

# Initial guess and bounds for variables
initial_guess = [0.1, 1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)]
bounds = [(-10, 10), (-1, 1), (-1, 1), (-1, 1)]  # Adjust bounds for alpha, a, b, c

# Constraint for scipy minimize
norm_constraint = {'type': 'eq', 'fun': constraint}

# Maximize alpha
max_alpha_result = minimize(
    lambda x: -objective(x),  # Negative because we want to maximize
    initial_guess,
    bounds=bounds,
    constraints=[norm_constraint],
    method='SLSQP',
)

# Minimize alpha
min_alpha_result = minimize(
    objective,  # Minimize directly
    initial_guess,
    bounds=bounds,
    constraints=[norm_constraint],
    method='SLSQP',
)

# Display results
print("Maximized alpha:", -max_alpha_result.fun)
print("Optimal values for max alpha:", max_alpha_result.x)

print("Minimized alpha:", min_alpha_result.fun)
print("Optimal values for min alpha:", min_alpha_result.x)
