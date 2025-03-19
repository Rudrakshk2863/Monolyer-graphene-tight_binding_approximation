import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
import pandas as pd

vpp_pi = -2.4  # Potential of pi bond in ev
e2p = 0
dim = 2  # dimension of hamiltonian
H = np.zeros([dim, dim], dtype=np.complex_)  # Hamiltonian matrix
Energy, Energy_ab, Energy2, Energy_ab2 = [], [], [], []
K_x, K_y, K_x2, K_y2 = [], [], [], []
a = 1.42  # inter-atomic distance in angstrom

# position vectors
R1 = [-a, 0]  # P.V. of neighbouring atom 1
R2 = [a / 2, (-a * np.sqrt(3)) / 2]  # P.V. of neighbouring atom 2
R3 = [a / 2, (a * np.sqrt(3)) / 2]  # P.V. of neighbouring atom 3
RA = [0.0, 0.0]  # P.V. of atom

diag = complex(e2p, 0)  # diagonal element of Hamiltonian matrix
iota = complex(0, 1)

# k-vectors
k1x_max = 2 * np.pi / (3 * a)  # K1-x Vector
k1y_max = 2 * np.pi / (3 * np.sqrt(3) * a)  # K1-y Vector

k2x_max = 2 * np.pi / (3 * a)  # K2-x Vector
k2y_max = -2 * np.pi / (3 * np.sqrt(3) * a)  # K2-y Vector

n = 100  # number of nodes

for i in range(3 * n):
    # gamma-k direction
    kx = k1x_max * (i / n)
    ky = k1y_max * (i / n)
    K_y.append(ky)
    K_x.append(kx)
    # gamma-k_dash direction
    kx2 = k2x_max * (i / n)
    ky2 = k2y_max * (i / n)
    K_x2.append(kx2)
    K_y2.append(ky2)

K_x = np.array(K_x)
K_y = np.array(K_y)
K_x2 = np.array(K_x2)
K_y2 = np.array(K_y2)

# k.r dot product

# gamma-k direction
kr1 = (K_x * (RA[0] - R1[0]) + K_y * (RA[1] - R1[1]))
kr2 = (K_x * (RA[0] - R2[0]) + K_y * (RA[1] - R2[1]))
kr3 = (K_x * (RA[0] - R3[0]) + K_y * (RA[1] - R3[1]))

w1 = iota * kr1
w2 = iota * kr2
w3 = iota * kr3

h_pi = np.exp(w1) + np.exp(w2) + np.exp(w3)  # energy associated with interaction of pi bonds

# gamma-k_dash direction
kr12 = (K_x2 * (RA[0] - R1[0]) + K_y2 * (RA[1] - R1[1]))
kr22 = (K_x2 * (RA[0] - R2[0]) + K_y2 * (RA[1] - R2[1]))
kr32 = (K_x2 * (RA[0] - R3[0]) + K_y2 * (RA[1] - R3[1]))

w12 = iota * kr12
w22 = iota * kr22
w32 = iota * kr32

h_pi2 = np.exp(w12) + np.exp(w22) + np.exp(w32)

# Solving for Energies
H[0, 0] = diag
H[1, 1] = diag
for i in h_pi:
    H[0, 1] = i * vpp_pi
    H[1, 0] = H[0, 1].conjugate()
    energies, vectors = LA.eigh(H)
    Energy.append(energies[0])  # Energy of bonding orbital in ground state
    Energy_ab.append(energies[1])  # Energy of anti-bonding orbital in ground state
for i in h_pi2:
    H[0, 1] = i * vpp_pi
    H[1, 0] = H[0, 1].conjugate()
    energies, vectors = LA.eigh(H)
    Energy2.append(energies[0])  # Same for gamma-k_dash direction
    Energy_ab2.append(energies[1])

x_axis = np.arange(0, len(h_pi), 1)
fig, ax = plt.subplots()
ax.plot(-x_axis, Energy2, color='blue', label='Energy Bonding')
ax.plot(-x_axis, Energy_ab2, color='red', label='Energy Anti-Bonding')
ax.plot(x_axis, Energy, color='blue')
ax.plot(x_axis, Energy_ab, color='red')
ax.set_xlabel('K-Vector')
ax.set_ylabel('Energy  (eV)')
ax.set_xticks([0, -(n + (n / 2)), -(n), n, n + (n / 2)])
ax.set_xticklabels(['$gamma$', 'M', 'K_dash', 'K', 'M'])
ax.legend()
ax.grid()
plt.show()

# Tabular storing of data

print("Gamma-k direction")
df_1 = pd.DataFrame({"Energy bonding": Energy, "K1x": K_x, "Energy anti-bonding": Energy_ab, "K1y": K_y})
df_1.to_csv(r'C:\Users\kailb\OneDrive\Desktop\gungun\TB01.dat', sep=' ')
print(df_1)

print("Gamma-k_dash direction")
df_2 = pd.DataFrame({"Energy bonding": Energy2, "K2x": K_x2, "Energy anti-bonding": Energy_ab2, "K2y": K_y2})
df_2.to_csv(r'C:\Users\kailb\OneDrive\Desktop\gungun\TB02.dat', sep=' ')
print(df_2)