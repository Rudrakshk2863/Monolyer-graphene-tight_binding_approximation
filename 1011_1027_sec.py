from __future__ import unicode_literals
import numpy as np
from scipy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt

vpp_pi = -2.4  # Potential of pi bond in ev
e2p = 0        #
dim = 2        # dimension of hamiltonian
H = np.zeros([dim, dim], dtype=np.complex_)      # Hamiltonian matrix
Energy, K_x, Energy_ab, K_y, K_x2, K_y2, Energy2, Energy_ab2 = [], [], [], [], [], [], [], []

a = 1.42       # inter-atomic distance in angstrom

# position vectors
R1 = [-a, 0]                          # P.V. of neighbouring atom 1
R2 = [a / 2, (-a * np.sqrt(3)) / 2]   # P.V. of neighbouring atom 2
R3 = [a / 2, (a * np.sqrt(3)) / 2]    # P.V. of neighbouring atom 3
RA = [0.0, 0.0]                       # P.V. of atom

diag = complex(e2p, 0)                # diagonal element of Hamiltonian matrix
iota = complex(0, 1)

# k-vectors
k1x_max = 2 * np.pi / (3 * a)                # K1-x Vector
k1y_max = 2 * np.pi / (3 * np.sqrt(3) * a)   # K1-y Vector

k2x_max = 2 * np.pi / (3 * a)                # K2-x Vector
k2y_max = -2 * np.pi / (3 * np.sqrt(3) * a)  # K2-y Vector

n = 10                                       # number of nodes

for i in range(3*n):
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

K_X = []
K_Y = []
K_X2 = []
K_Y2 = []
for i in K_x:
    for j in K_y:
        # gamma-k direction
        kr1 = (i * (RA[0] - R1[0]) + j * (RA[1] - R1[1]))
        kr2 = (i * (RA[0] - R2[0]) + j * (RA[1] - R2[1]))
        kr3 = (i * (RA[0] - R3[0]) + j * (RA[1] - R3[1]))

        w1 = iota * kr1; w2 = iota * kr2; w3 = iota * kr3

        h_pi = np.exp(w1) + np.exp(w2) + np.exp(w3)      # energy associated with interaction of pi bonds
        H[0, 0] = diag
        H[1, 1] = diag

        H[0, 1] = h_pi * vpp_pi
        H[1, 0] = H[0, 1].conjugate()
        energies, vectors = LA.eigh(H)
        Energy.append(energies[0])                      # Energy of bonding orbital in ground state
        Energy_ab.append(energies[1])
        K_X.append(i)
        K_Y.append(j)

# Same for gamma-K_dash direction

for i in K_x2:
    for j in K_y2:
        # gamma-k direction
        kr12 = (i * (RA[0] - R1[0]) + j * (RA[1] - R1[1]))
        kr22 = (i * (RA[0] - R2[0]) + j * (RA[1] - R2[1]))
        kr32 = (i * (RA[0] - R3[0]) + j * (RA[1] - R3[1]))

        w12 = iota * kr12; w22 = iota * kr22; w32 = iota * kr32

        h_pi2 = np.exp(w12) + np.exp(w22) + np.exp(w32)      # energy associated with interaction of pi bonds
        H[0, 0] = diag
        H[1, 1] = diag

        H[0, 1] = h_pi2 * vpp_pi
        H[1, 0] = H[0, 1].conjugate()
        energies, vectors = LA.eigh(H)
        Energy2.append(energies[0])                         # Energy of bonding orbital in ground state
        Energy_ab2.append(energies[1])                      # Energy of anti-bonding orbital in ground state
        K_X2.append(i)
        K_Y2.append(j)


# Tabulation of data

print("Gamma-k direction")
df_1 = pd.DataFrame({"Kx": K_X, "ky": K_Y, "Energy bonding": Energy, "Energy anti-bonding": Energy_ab, "Kx2": K_X2, "ky2": K_Y2, "Energy bonding2": Energy2, "Energy anti-bonding2": Energy_ab2})
df_1.to_csv(r'C:\Users\Rudraksh\PycharmProjects\pythonProject\secproject01.csv', sep=',')
print(df_1)

points = pd.read_csv('secproject01.csv')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = points['Kx'].values
y = points['ky'].values
z = points['Energy anti-bonding'].values
ax.scatter(x, y, z, c='r')
x2 = points['Kx'].values
y2 = points['ky'].values
z2 = points['Energy bonding'].values
ax.scatter(x2, y2, z2, c='b')
x3 = points['Kx2'].values
y3 = points['ky2'].values
z3 = points['Energy bonding2'].values
ax.scatter(x3, y3, z3, c='b')
x4 = points['Kx2'].values
y4 = points['ky2'].values
z4 = points['Energy anti-bonding2'].values
ax.scatter(x4, y4, z4, c='r')
# plt.contourf(x4, y4, z4, 20, cmap='RdGy')
# plt.colorbar()
# plt.contourf(X, Y, Z, 20, cmap='RdGy')
# plt.colorbar();
# plt.contourf(X, Y, Z, 20, cmap='RdGy')
# plt.colorbar();
# plt.contourf(X, Y, Z, 20, cmap='RdGy')
# plt.colorbar();
plt.show()

