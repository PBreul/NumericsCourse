import os
# This script calls the Main Program with different courant numbers
c_array = (0.1, 0.2, 0.5, 0.8, 0.9, 1.0, 1.5)

for c in c_array:
    os.system("python3 LinearAdvection.py {}".format(c))
