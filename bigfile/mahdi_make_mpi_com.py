import ctypes
from mpi4py import MPI

# Determine the appropriate MPI_Comm type
if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    mpi_comm_type = 'int'
else:
    mpi_comm_type = 'void_p'

# Write the Cython code to a .pxi file
with open('mpi_comm_config.pxi', 'w') as f:
    f.write(f'cdef {mpi_comm_type} MPI_Comm\n')
