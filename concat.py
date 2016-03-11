# to run
# salloc -N 1 --mem-per-cpu=3500 -p debug -t 30 --qos=premium
# module load h5py mpi4py
# srun -c 1 -n 24 --mem-per-cpu=3500 python-mpi -u ./concat2.py

from mpi4py import MPI
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import time

rank = MPI.COMM_WORLD.Get_rank()
numprocs = MPI.COMM_WORLD.Get_size()

datapath = "/global/cscratch1/sd/gittens/hdf5-export/data"
basefname = "chunkOut"
filelist = [fname for fname in listdir(datapath) if fname.startswith(basefname)]

if rank == 0:
    print "%s : Starting processes" % time.asctime(time.localtime())

myfiles = [fname for (index, fname) in enumerate(filelist) if (index % numprocs == rank)]
myhandles = [None]*len(myfiles) # a list of file handles for the files this process will read from
myrowindices = [None]*len(myfiles) # a list containing the list of indices for each of the files this process will read from
mynumrows = 0
for (idx, fname) in enumerate(myfiles):
    myhandles[idx] = h5py.File(join(datapath, fname), "r")
    mynumrows = mynumrows + myhandles[idx].get("values").shape[0]
    buf = np.empty((myhandles[idx].get("indices").size,), dtype=np.int32)
    myhandles[idx].get("indices").read_direct(buf)
    myrowindices[idx] = buf.tolist()

mynumrows = MPI.COMM_WORLD.gather(mynumrows, root=0)
if rank == 0:
    numRows = sum(mynumrows)
else:
    numRows = None
numRows = MPI.COMM_WORLD.bcast(numRows, root=0)
numCols = myhandles[0].get("values").shape[1]

# for efficiency (?) set the chunkSize to be the same as the chunks we're writing out, and use highest level of compression because 
# this file is huge
chunkSize = 4000
if rank == 0:
    fout = h5py.File(join(datapath, "oceanTemps.hdf5"), "w")
    temperature = fout.create_dataset("temperatures", (numRows, numCols), dtype=np.float64, compression=9, chunks=(chunkSize, numCols))

startrow = 0
tempbuf = np.zeros((chunkSize, numCols), dtype=np.float64)
while startrow < numRows:
    endrow = min(startrow + chunkSize, numRows)
    rowRange = set(np.arange(startrow, endrow)) # all rows including startrow, excluding endrow
    if rank == 0:
        print "%s : Processing rows %d--%d" % (time.asctime(time.localtime()), startrow, endrow-1)

    # find the indices from each process that are within the rowRange, and the corresponding rows,
    # then pass those back to the root for writing out
    foundIndices = []
    foundRows = []
    for (fhIndex, fh) in enumerate(myhandles):
        intersect = list( set(myrowindices[fhIndex]) & rowRange)
        if len(intersect) > 0:
            foundIndices.extend(intersect)
            rowIndexOffsets = [myrowindices[fhIndex].index(rowindex) for rowindex in intersect]
            fh.get("values").read_direct(tempbuf, dest_sel=np.s_[0:fh.get("indices").size, :])
            foundRows.extend([tempbuf[offset, :] for offset in rowIndexOffsets])

    foundIndices = MPI.COMM_WORLD.gather(foundIndices, root=0)
    foundRows = MPI.COMM_WORLD.gather(foundRows, root=0)

    if rank == 0:
        allIndices = [index for sublist in foundIndices for index in sublist]
        allRows = [row for sublist in foundRows for row in sublist]

        orderOffsets = np.argsort(allIndices)
        sortedIndices = [allIndices[offset] for offset in orderOffsets]
        sortedRows = [allRows[offset] for offset in orderOffsets]

        temperature[startrow:endrow, :] = np.array(sortedRows)
        print "%s : Wrote rows %d--%d" % (time.asctime(time.localtime()), sortedIndices[0], sortedIndices[-1])

    # MPI.COMM_WORLD.barrier()
    startrow = endrow

for handle in myhandles:
    handle.close()

if rank == 0:
    fout.close()
