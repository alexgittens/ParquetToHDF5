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
import math

rank = MPI.COMM_WORLD.Get_rank()
numProcs = MPI.COMM_WORLD.Get_size()
numWriters = 10
writerRanks = list(np.arange(numWriters))
chunkSize = 4000


datapath = "/global/cscratch1/sd/gittens/hdf5-export/data"
basefname = "chunkOut"
filelist = [fname for fname in listdir(datapath) if fname.startswith(basefname)]

if rank == 0:
    print "%s : Starting processes" % time.asctime(time.localtime())

myfiles = [fname for (index, fname) in enumerate(filelist) if (index % numProcs == rank)]
myhandles = [None]*len(myfiles) # a list of file handles for the files this process will read from
myrowindices = [None]*len(myfiles) # a list containing the list of indices for each of the files this process will read from
mynumrows = 0
for (idx, fname) in enumerate(myfiles):
    myhandles[idx] = h5py.File(join(datapath, fname), "r")
    mynumrows = mynumrows + myhandles[idx].get("values").shape[0]
    buf = np.empty((myhandles[idx].get("indices").size,), dtype=np.int32)
    myhandles[idx].get("indices").read_direct(buf)
    myrowindices[idx] = buf.tolist()
mymaxrownum = max([len(l) for l in myrowindices])

mynumrows = MPI.COMM_WORLD.gather(mynumrows, root=0)
if rank == 0:
    numRows = sum(mynumrows)
else:
    numRows = None
numRows = MPI.COMM_WORLD.bcast(numRows, root=0)
numCols = myhandles[0].get("values").shape[1]

# DEBUGGING
numRows = 9510
chunkSize = 200


# set the chunkSize to be the same as the chunks we're writing out
fout = h5py.File(join(datapath, "oceanTemps.hdf5"), "w", driver="mpio", comm=MPI.COMM_WORLD)
temperature = fout.create_dataset("temperatures", (numRows, numCols), dtype=np.float64, chunks=(chunkSize, numCols))

startrow = 0
while startrow < numRows:
    endrow = min(startrow + chunkSize*numWriters, numRows)
    rowRange = set(np.arange(startrow, endrow)) # all rows including startrow, excluding endrow
    if rank == 0:
        print "%s : Processing rows %d--%d" % (time.asctime(time.localtime()), startrow, endrow-1)

    # find the indices from each process that are within the rowRange, and the corresponding rows
    foundIndices = []
    foundRows = []
    writeBuf = np.zeros((mymaxrownum, numCols), dtype=np.float64)
    for (fhIndex, fh) in enumerate(myhandles):
        intersect = sorted(list( set(myrowindices[fhIndex]) & rowRange))
        if len(intersect) > 0:
            foundIndices.extend(intersect)
            rowIndexOffsets = [myrowindices[fhIndex].index(rowindex) for rowindex in intersect]
            fh.get("values").read_direct(writeBuf, dest_sel=np.s_[0:fh.get("indices").size, :])
            foundRows.extend([writeBuf[offset, :] for offset in rowIndexOffsets])
    if rank == 0:
        print "%s : done searching for relevant rows" % time.asctime(time.localtime())

    # the writers do chunkSize aligned output to avoid IO contention
    # divide up the indices and rows so they are assigned to the correct writer processes
    numWritersNeeded = int(math.ceil(len(rowRange)/float(chunkSize)))
    chunkOffsets = list(np.arange(startrow, endrow, chunkSize))
    indicesForWriter = []
    rowsForWriter = []
    for writerIdx in np.arange(numWritersNeeded):
        writerOffsets = [foundIndices.index(rowIdx) for rowIdx in 
                           filter(lambda rowIdx: (rowIdx >= chunkOffsets[writerIdx] and rowIdx < chunkOffsets[writerIdx] + chunkSize), foundIndices)]
        indicesForWriter.append([foundIndices[rowOffset] for rowOffset in writerOffsets])
        rowsForWriter.append([foundRows[rowOffset] for rowOffset in writerOffsets])
    if rank == 0:
        print "%s : done assigning rows to writers" % time.asctime(time.localtime())

    for writerIdx in np.arange(numWritersNeeded):
        indicesForWriter[writerIdx] = MPI.COMM_WORLD.gather(indicesForWriter[writerIdx], root=writerRanks[writerIdx])
        rowsForWriter[writerIdx] = MPI.COMM_WORLD.gather(rowsForWriter[writerIdx], root=writerRanks[writerIdx])
    if rank == 0:
        print "%s : done gathering rows to writers" % time.asctime(time.localtime())

    if rank in writerRanks[0:numWritersNeeded]:
        writerRankIndex = writerRanks.index(rank)
        allIndices = [index for sublist in indicesForWriter[writerRankIndex] for index in sublist]
        allRows = [row for sublist in rowsForWriter[writerRankIndex] for row in sublist]

        if len(allIndices) > 0:
            orderOffsets = np.argsort(allIndices)
            sortedIndices = [allIndices[offset] for offset in orderOffsets]
            sortedRows = [allRows[offset] for offset in orderOffsets]

            writerstartrow = sortedIndices[0]
            writerendrow = sortedIndices[-1] + 1
            temperature[writerstartrow:writerendrow, :] = np.array(sortedRows)
        print "%s : done writing from process %d" % (time.asctime(time.localtime()), rank)

    MPI.COMM_WORLD.barrier()
    if rank == 0:
        print "%s : Wrote rows %d--%d" % (time.asctime(time.localtime()), startrow, endrow-1)

    startrow = endrow

for handle in myhandles:
    handle.close()

fout.close()
