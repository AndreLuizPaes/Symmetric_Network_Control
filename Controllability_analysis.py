
import numpy as np
import math as math
from scipy.sparse import eye, csr_matrix, kron
from scipy.linalg import null_space, eigvals, expm
from itertools import permutations
from numpy.linalg import matrix_rank, det

n = 8
dim = int(math.comb(n + 3, n)) - 1

# Define Pauli matrices (including identity)
sigma = [
    eye(2, format='csr'),
    csr_matrix(np.array([[0, 1], [1, 0]])),
    csr_matrix(np.array([[0, -1j], [1j, 0]])),
    csr_matrix(np.array([[1, 0], [0, -1]]))
]

xyz = np.array([0, 0, 0])
xyz_all = np.zeros((dim, 3), dtype=int)
basis = [csr_matrix((2**n, 2**n)) for _ in range(dim)]

system_op = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, 2]
])

# Generate binary states
State_ids = np.ones((2**n, n), dtype=int)
for i in range(1, 2**n):
    State_hold = State_ids[i - 1].copy()
    if State_hold[-1] == 1:
        State_hold[-1] = 0
    else:
        finder = False
        id_search = n - 2
        while not finder and id_search >= 0:
            if State_hold[id_search] == 1:
                State_hold[id_search+1:] = [1] * (n - id_search-1)
                State_hold[id_search] = 0
                finder = True
            id_search -= 1
    State_ids[i] = State_hold
State_count = State_ids.sum(axis=1)
T = np.zeros((n + 1, 2**n), dtype=float)
for i in range(n, -1, -1):
    T_hold = (State_count == i).astype(float)
    norm = np.sqrt(np.sum(T_hold))
    T[n - i] = T_hold / norm if norm != 0 else T_hold

null_T = null_space(T)
T = np.vstack((T, null_T.T))
T = csr_matrix(T)
cT = T.getH()
# Build the operator basis
for i in range(dim):
    if xyz[2] == n or (xyz.sum() == n and xyz[2] > 0):
        xyz[1] += 1
        xyz[2] = 0
    elif xyz[1] == n or (xyz.sum() == n and xyz[2] == 0):
        xyz[0] += 1
        xyz[1] = 0
    else:
        xyz[2] += 1

    permuts = [1] * (n - xyz.sum()) + [2] * xyz[0] + [3] * xyz[1] + [4] * xyz[2]
    unique_perms = np.unique(list(permutations(permuts)), axis=0)

    for perm in unique_perms:
        sigma_iter = sigma[perm[0] - 1]
        for s in perm[1:]:
            sigma_iter = kron(sigma_iter, sigma[s - 1])
        basis[i] += sigma_iter

    basis[i] = T @ basis[i] @ cT
    basis[i] = basis[i][:n + 1, :n + 1]
    xyz_all[i] = xyz

# Check if Lie algebra spans full su(n+1)
dim_Lie = (n + 1)**2
generated_Lie = [np.zeros((n + 1, n + 1), dtype=complex) for _ in range(dim_Lie)]
vector_Lie = [np.zeros(((n + 1)**2, 1), dtype=complex) for _ in range(dim_Lie)]

system_Lie = []
dim_generated = system_op.shape[0]

for i in range(dim_generated):
    idx = np.where(np.all(xyz_all == system_op[i], axis=1))[0][0]
    generated_Lie[i] = basis[idx].toarray()
    vector_Lie[i] = generated_Lie[i].reshape(-1, 1)
    system_Lie.append(vector_Lie[i])

system_Lie_mat = np.hstack(system_Lie)
rank_system = matrix_rank(system_Lie_mat)

n_sub = 0
n_master = 1

while n_master < dim_generated and dim_generated < dim_Lie:
    commutator = generated_Lie[n_master] @ generated_Lie[n_sub] - generated_Lie[n_sub] @ generated_Lie[n_master]
    vector_candidate = commutator.reshape(-1, 1)

    independ_ind = True
    if np.allclose(commutator, 0) or matrix_rank(np.hstack([system_Lie_mat, vector_candidate])) == rank_system:
        independ_ind = False
    if n_master not in [1, 2] and n_sub not in [1, 2]:
        independ_ind = False

    if independ_ind:
        scale_factor = 1j / np.max(np.abs(eigvals(commutator)))
        new_operator = scale_factor * commutator
        generated_Lie[dim_generated] = new_operator
        vector_Lie[dim_generated] = new_operator.reshape(-1, 1)
        system_Lie_mat = np.hstack([system_Lie_mat, vector_Lie[dim_generated]])
        rank_system = matrix_rank(system_Lie_mat)
        dim_generated += 1

    if n_sub == n_master - 1:
        n_master += 1
        n_sub = 0
    else:
        n_sub += 1

if dim_generated == (n + 1)**2:
    print("The system is subspace controllable on the Lie group U(n+1), since the dimension of the Lie Algebra generated equals (n+1)^2")
elif dim_generated == (n + 1)**2-1:
    print("The system is subspace controllable on the Lie group SU(n+1), since the dimension of the Lie Algebra generated equals (n+1)^2-1")
else:
    print("System is NOT subspace controllable, since the dimension of the Lie Algebra generated is SMALLER than (n+1)^2")

unitary_check = 1;
determinant_check = 1;
for iter in range(1,(n+1)**2):
    expA = expm(1j*generated_Lie[iter-1])
    ActA = expA @ (csr_matrix(expA).getH()).toarray()
    if np.max(np.abs(eigvals(ActA-eye(n+1)))) >=1e-10:
        print("The Lie group have elements that are not unitary")
        unitary_check=0
    if np.abs(det(ActA)-1) >= 1e-10:
        print("The Lie group have elements that does not have determinant equal to 1")
        determinant_check=0
    
if unitary_check == 1:
    print("The Lie group is unitary")

if determinant_check == 1:
    print("The Lie group only has elements with determinant equal to 1")