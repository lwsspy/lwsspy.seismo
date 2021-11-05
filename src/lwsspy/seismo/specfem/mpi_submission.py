#!/usr/bin/env python

from subprocess import Popen, PIPE, STDOUT

from time import sleep
from mpi4py import MPI
from ...utils import Timer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# Important for specfem simulations
p = Popen(["/bin/echo", "Hello", "from", "rank",
           f"{rank}/{nprocs}"], stdout=PIPE, stderr=PIPE, text=True)
p.wait()
out, err = p.communicate()
print(out, err)


if rank == 0:
    # Important for specfem simulations
    print(f"Running Specfem from rank: {rank}/{nprocs}.", flush=True)
    with Timer():
        p = Popen(["jsrun", "-n", "6", "-a", "4", "-c", "4", "-g", "1", "./bin/xspecfem3D"],
                  stdout=PIPE, stderr=PIPE, text=True)
        p.wait()
        out, err = p.communicate()
        print(out, err)
else:
    sleep(float(rank))
    print(f"Not running specfem on {rank}/{nprocs}", flush=True)

# Barrier
comm.Barrier()

if rank != 0:
    print(f"'Ha! I waited', said {rank} of {nprocs}.")


print("Done. {rank}/{nproc}")
