import os

# This script calls the Main Program with different courant numbers
print("Calculate time evolution of Gaussian")
# number of grid points
nx = 250

# number of time steps
nt = 560

# Courant number
c = 0.2

# initial condition
initial_cond = "gauss"

print("c", c)
os.system("python3 TimeEvolutionObservation.py {} {} {} {}".format(nx, nt, c, initial_cond))

c = 1.4
nx = 125
nt = 40

print("c", c)
os.system("python3 TimeEvolutionObservation.py {} {} {} {}".format(nx, nt, c, initial_cond))

print("Calculate time evolution of Step Function")

initial_cond = "step"
nx = 250
nt = 560
c = 0.2

print("c", c)
os.system("python3 TimeEvolutionObservation.py {} {} {} {}".format(nx, nt, c, initial_cond))

print("Make accuracy plots, that might take a second")

# os.system("python3 AccuracyInvestigation.py")
