import h5py
import numpy as np
from os.path import join
from os import listdir

hdf5file = "/global/cscratch1/sd/gittens/hdf5-export/data/oceanTemps.hdf5"
basedir = '/global/cscratch1/sd/gittens/hdf5-export'
basefname = 'oceanTempsChunk'
filelist = [fname for fname in listdir(basedir) if fname.startswith(basefname)]

fin = h5py.File(hdf5file, "r")
temps = fin.get("temperatures")
numRows = temps.shape[0]
numSamples = 50
indices = sorted(list(set(np.random.randint(numRows, size=numSamples)) | set([0, temps.shape[0]-1])))

h5rows = np.array(temps[indices, :])
fin.close()

startrows = map(lambda fname: int(fname[15:-3]), filelist)
fileindices = []
fileoffsets = []
for index in indices:
    startingRow = sorted(filter(lambda startrow: startrow <= index, startrows))[-1]
    fileindices.append(startrows.index(startingRow))
    fileoffsets.append(index - startingRow)

splitrows = np.empty_like(h5rows)
for offsetidx in np.arange(h5rows.shape[0]):
    fin = h5py.File(join(basedir, filelist[fileindices[offsetidx]]), "r")
    splitrows[offsetidx, :] = fin.get("temperatures")[fileoffsets[offsetidx], :]
    fin.close()

print np.linalg.norm(h5rows - splitrows)

