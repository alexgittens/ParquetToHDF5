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

# open all the files on the separate processes (to cache descriptors to avoid
# metadata latencies)

if rank == 0:
    print "%s : Starting processes" % time.asctime(time.localtime())
#print "starting process %d/%d" % (rank + 1, numprocs)

myfiles = [fname for (index, fname) in enumerate(filelist) if (index % numprocs == rank)]
myhandles = []
myrowindices = []
mynumrows = 0
for fname in myfiles:
    myhandles.append(h5py.File(join(datapath, fname), "r"))
    mynumrows = mynumrows + myhandles[-1].get("values").shape[0]
    buf = np.empty((myhandles[-1].get("indices").size,), dtype=np.int32)
    myhandles[-1].get("indices").read_direct(buf)
    myrowindices.append(buf.tolist())
mynumrows = MPI.COMM_WORLD.gather(mynumrows, root=0)
if rank == 0:
    numRows = sum(mynumrows)
else:
    numRows = None
numRows = MPI.COMM_WORLD.bcast(numRows, root=0)
numCols = myhandles[0].get("values").shape[1]

if rank == 0:
    fout = h5py.File(join(datapath, "oceanTemps.hdf5"), "w")
    temperature = fout.create_dataset("temperatures (K)", (numRows, numCols), dtype=np.float64)

## the root collects rows in chunks from the other processes, then writes
## them in consecutive order to the output file

startrow = 0
chunkSize = 4000
tempbuf = np.zeros((chunkSize, numCols), dtype=np.float64)
while startrow < numRows:
    # range contains chunkSize elements
    endrow = min(startrow + chunkSize, numRows)
    rowRange = set(np.arange(startrow, endrow)) # all rows including startrow, excluding endrow
    if rank == 0:
        print "%s : Processing rows %d--%d" % (time.asctime(time.localtime()), startrow, endrow-1)

    # assign rows and indices from each process
    foundIndices = []
    foundRows = []
    for (fhIndex, fh) in enumerate(myhandles):
        intersect = list( set(myrowindices[fhIndex]) & rowRange)
        if len(intersect) > 0:
            foundIndices.extend(intersect)
            rowIndexOffsets = [myrowindices[fhIndex].index(rowindex) for rowindex in intersect]
            fh.get("values").read_direct(tempbuf, dest_sel=np.s_[0:fh.get("indices").size, :])
            foundRows.extend([tempbuf[offset, :] for offset in rowIndexOffsets])
        # print "file %d/%d searched on processor %d" % (fhIndex, len(myhandles), rank)
        #print "matched %d/%d indices on processor %d" % (len(foundIndices), chunkSize, rank)

    #if rank == 0:
        # print "%s : Collecting rows" % time.asctime(time.localtime())
    foundIndices = MPI.COMM_WORLD.gather(foundIndices, root=0)
    foundRows = MPI.COMM_WORLD.gather(foundRows, root=0)

    if rank == 0:
        # print "%s : Flattening rows" % time.asctime(time.localtime())
        allIndices = [index for sublist in foundIndices for index in sublist]
        allRows = [row for sublist in foundRows for row in sublist]

        # print "%s : Sorting indices" % time.asctime(time.localtime())
        orderOffsets = np.argsort(allIndices)
        sortedIndices = [allIndices[offset] for offset in orderOffsets]
        # print "%s : Sorting rows" % time.asctime(time.localtime())
        sortedRows = [allRows[offset] for offset in orderOffsets]

        # print "%s : Writing out rows" % time.asctime(time.localtime())
        temperature[startrow:endrow, :] = np.array(sortedRows)
        print sortedIndices
    MPI.COMM_WORLD.barrier()
    startrow = endrow

for handle in myhandles:
    handle.close()

if rank == 0:
    fout.close()
