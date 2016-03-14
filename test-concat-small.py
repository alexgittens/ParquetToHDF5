from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import h5py
import numpy as np

conf = (SparkConf()).setAppName("testHDF5ConversionSmall")
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

oceanTempsParquetFile = "/global/cscratch1/sd/gittens/CFSROparquet/finalmat.parquet"
hdf5file = "/global/cscratch1/sd/gittens/hdf5-export/data/oceanTemps.hdf5"

fin = h5py.File(hdf5file, "r")
temps = fin.get("temperatures")
numRows = temps.shape[0]
indices = list(np.arange(numRows))

h5rows = np.array(temps)

df = sqlContext.read.load(oceanTempsParquetFile)
data = sorted(df.rdd.filter(lambda row: row[0] in indices).collect(), key= lambda pair: pair[0])
sparkrows = np.array(map(lambda pair: pair[1], data))

print np.linalg.norm(h5rows - sparkrows)

