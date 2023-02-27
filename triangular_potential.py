# ----------------------------
# Triangular potential
# ----------------------------
# Finite differences method as developed by Truhlar JCP 10 (1972) 123-132
#
# code by Jordi Faraudo
# edited by Marc Rodriguez
#
#
import numpy as np
import matplotlib.pyplot as plt

#Ask the user for the value for initial potential (V_0):
print("Enter desired launch slope from the triangular potential in eV/nm (recommended 1000 eV/nm):")
vo=float(input())*0.00194469

#Potential as a function of position
def getV(x):
    potvalue = vo*abs(x)
    return potvalue

#Discretized Schrodinger equation in n points (FROM 0 to n-1)
def Eq(n,h,x):
    F = np.zeros([n,n])
    for i in range(0,n):
        F[i,i] = -2*((h**2)*getV(x[i]) + 1)
        if i > 1:
           F[i,i-1] = 1
           if i < n-2:
              F[i,i+1] = 1
    return F

#-------------------------
# Main program
#-------------------------
# Interval for calculating the wave function [-L/2,L/2]
L =8 
xlower = -L/2.0
xupper = L/2.0

#Discretization options
h = 0.02  #discretization in space

#Create coordinates at which the solution will be calculated
x = np.linspace(xlower,xupper,int(L/h))

#grid size (how many discrete points to use in the range [-L/2,L/2])
npoints=len(x)

print("Using",npoints, "grid points.")

#Calculation of discrete form of Schrodinger Equation
print("Calculating matrix...")
F=Eq(npoints,h,x)

#diagonalize the matrix F
print("Diagonalizing...")
eigenValues, eigenVectors = np.linalg.eig(F)

#Order results by eigenvalue
# w ordered eigenvalues and vs ordered eigenvectors
idx = eigenValues.argsort()[::-1]   
w = eigenValues[idx]
vs = eigenVectors[:,idx]

#Energy Level
e=-w/(2.0*h**2)
E =-27.2114*w/(2.0*h**2)

#Print Energy Values
print("RESULTS:")
for k in range(0,6):
	print("State ",k+1," Energy = %.2f" %e[k]+" hartrees"+" = %.2f" %E[k]+" eV")

#Init Wavefunction (empty list with npoints elements)
psi = [None]*npoints

#Calculation of normalised Wave Functions
for k in range(0,len(w)):
	psi[k] = vs[:,k]
	integral = h*np.dot(psi[k],psi[k])
	psi[k] =psi[k]/(integral**0.5)

#Plot Wave functions
print("Plotting")

#v = int(input("\n Quantum Number (enter 0 for ground state):\n>"))

for v in range(0,1):
    plt.plot(x*0.052,20*abs(psi[v])+E[v],label=r'$\psi_v(x)$, k = ' + str(v+1))
    plt.plot(x*0.052,70*(psi[v])**2,label='Probabilitat')
    plt.plot(x*0.052, 27.2*getV(x),label='Energia potencial $V(x)$')
    plt.hlines(E[v],xmin=-0.2, xmax= 0.2, color='red', linestyle='dashed', label='Nivell energètic '+str(v+1))
    plt.title(r'$n=$'+ str(v+1) + r', $E_n$=' + '{:.2f}'.format(E[v])+'eV')
    plt.legend()
    plt.xlabel(r'$x$(nm)')
    plt.ylabel(r'$E(eV)$')
    plt.show()

for v in range(1,4):
    plt.plot(x*0.052,20*psi[v]+E[v],label=r'$\psi_v(x)$, k = ' + str(v+1))
    plt.plot(x*0.052,70*(psi[v])**2,label='Probabilitat')
    plt.plot(x*0.052, 27.2*getV(x),label='Energia potencial $V(x)$')
    plt.hlines(E[v],xmin=-0.2, xmax= 0.2, color='red', linestyle='dashed', label='Nivell energètic '+str(v+1))
    plt.title(r'$n=$'+ str(v+1) + r', $E_n$=' + '{:.2f}'.format(E[v])+'eV')
    plt.legend()
    plt.xlabel(r'$x$(nm)')
    plt.ylabel(r'$E(eV)$')
    plt.show()

print("Bye")
