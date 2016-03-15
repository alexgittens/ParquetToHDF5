import math, h5py, time, numpy as np
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
numProcs = MPI.COMM_WORLD.Get_size()

def report(status):
    print "%s : %d/%d : %s" % (time.asctime(time.localtime()), rank + 1, numProcs, status) 

def reportroot(status):
    if rank == 0:
        report(status)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

chunkSize = 5000
inFname = "data/oceanTemps.hdf5"
baseOutFname = "outChunk"

report("Opening file for read in")
fin = h5py.File(inFname, "r")
temps = fin.get("temperatures")
numRows = temps.shape[0]
numCols = temps.shape[1]

report("Assigning indices")
indexChunks = list(chunks(np.arange(numRows), chunkSize))
myIndexChunkOffset = [idx for idx in np.arange(len(indexChunks)) if idx % numProcs == rank]
myNumChunks = len(myIndexChunkOffset)

for (idxCounter, idx) in enumerate(myIndexChunkOffset):
    startRowIndex = indexChunks[idx][0]
    endRowIndex = indexChunks[idx][-1]

    report("Creating output file for chunk %d/%d of my input" % (idxCounter + 1, myNumChunks))
    myOutFile = h5py.File(baseOutFname + str(startRowIndex) + ".h5", "w")
    myOutTemps = myOutFile.create_dataset("temperatures", (endRowIndex - startRowIndex + 1, numCols), dtype=np.float64, compression="gzip", compression_opts=9)

    report("Reading chunk %d/%d of my input" % (idxCounter + 1, myNumChunks))
    data = temps[startRowIndex:(endRowIndex+1), :]

    report("Writing compressed output chunk %d/%d" % (idxCounter + 1, myNumChunks))
    myOutTemps[...] = data

    myOutFile.close()

fin.close()
MPI.COMM_WORLD.barrier()
reportroot("Done!")

