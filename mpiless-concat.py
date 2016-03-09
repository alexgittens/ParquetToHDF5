import h5py
import numpy as np
from os import listdir
from os.path import isfile, join

datapath = "/global/cscratch1/sd/gittens/hdf5-export/data"
basefname = "chunkOut"

fout = h5py.File(join(datapath, "serial-oceanTemps.hdf5"), "w")

print "Computing the data size"
filelist = [fname for fname in listdir(datapath) if fname.startswith(basefname)]
numCols = h5py.File(join(datapath, filelist[0]), "r").get("values").shape[1]
numRows = 0
for fname in filelist:
    numRows += h5py.File(join(datapath, fname), "r").get("values").shape[0]
print "(numRows, numCols) = (%d, %d)" % (numRows, numCols)

temperatures = fout.create_dataset('temperatures', (numRows, numCols), dtype=np.float64)

for (findex, fname) in enumerate(filelist):
    print "working on file %d/%d" % (findex + 1, len(filelist))
    fin = h5py.File(join(datapath, fname), "r")
    indices = fin.get("indices")
    values = fin.get("values")
    for index in np.arange(indices.size):
        temperatures[indices[index], :] = values[index, :]
    fin.close()

fout.close()
