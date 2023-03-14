
from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
number_of_processes = comm.Get_size()
rank = comm.Get_rank()
root = 0
size = number_of_processes-1

def Nsendall(array):
    for i in range(1,number_of_processes):
        # print(i)
        comm.send(array, dest=i,tag=i)
        # print('Process {} sent data:'.format(rank),array)
    comm.Barrier()
start = MPI.Wtime()
if rank==root:
    arr = np.arange(10**3)
    # chunks_of_arr = np.array_split(arr, number_of_processes-1)
    # print("Splitted Array: ",chunks_of_arr)
    # print('sent array:',chunks_of_arr)
    sent_data = Nsendall(arr)
    # print('sent array:',chunks_of_arr)
else:
    received_arr = comm.recv(source=root)
    # print('Process {} received data:'.format(rank),received_arr)
    comm.Barrier()
end = MPI.Wtime()
total_time= end-start
print(f" R:{rank} Time:{total_time}")
