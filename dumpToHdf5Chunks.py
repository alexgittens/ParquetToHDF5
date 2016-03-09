# To run:
#  salloc -N 7 -p debug -t 00:30:00 --ccm
#  bash
#  modue load h5py
#  module load spark
#  start-all.sh
#  pyspark --master $SPARKURL --driver-memory 15G --executor-memory 32G
 
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import h5py
import numpy as np

conf = (SparkConf()).setAppName("HDF5Exporter")
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

oceanTempsParquetFile = "/global/cscratch1/sd/gittens/CFSROparquet/finalmat.parquet"
basefname = "/global/cscratch1/sd/gittens/hdf5-export/data/chunkOut"

def chunkWriter(chunkIndex, rowIter):
    listOfRows = [row for row in rowIter]
    numRows = len(listOfRows)
    rowMatrix = np.empty([numRows, listOfRows[0][1].size], dtype=np.float64)
    indices = np.empty((numRows,), dtype=np.int32)
    nextRowIndex = 0
    for row in listOfRows:
        indices[nextRowIndex] = row[0]
        rowMatrix[nextRowIndex, :] = row[1]
        nextRowIndex += 1 
    f = h5py.File(basefname + str(chunkIndex) + ".h5", "w")
    indexDset = f.create_dataset("indices", data=indices)
    valsDset = f.create_dataset("values", data=rowMatrix)
    f.close()
    return iter([len(indices)])

df = sqlContext.read.load(oceanTempsParquetFile)
# need to repartition?
# df.rdd.getNumPartitions() = 2880
df2 = df.rdd.mapPartitionsWithIndex(chunkWriter)
df2.count()

