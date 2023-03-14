from mpi4py import MPI
import numpy as np
import math

comm = MPI.COMM_WORLD
number_of_processes = comm.Get_size()
rank = comm.Get_rank()
root = 0
arr = np.array([])
size = number_of_processes-1



def Esendall(data):
    destA = 2*rank + 1
    destB = 2*rank + 2
    if destA<=size:
        comm.send(data,dest =destA)
        # print('Process {} sent data:'.format(rank),data)
    if destB<=size:
        comm.send(data,dest = destB)
        # print('Process {} sent data:'.format(rank),data)
    comm.Barrier()

start = MPI.Wtime()
if rank==root:
    arr = np.arange(10**7)
    # arr1 = np.array_split(arr, number_of_processes)
    # sent_data = Esendall(arr)
else:
    recvProc = int((rank-1)/2)
    # print("hello my rank is:",rank)
    # if recvProc <= number_of_processes:
    arr=comm.recv(source = recvProc)
    # print('Process {} received data:'.format(rank),arr)
    # comm.Barrier()


Esendall(arr)
end = MPI.Wtime()
total_time = end - start
print(f" R:{rank} Time:{total_time}")
