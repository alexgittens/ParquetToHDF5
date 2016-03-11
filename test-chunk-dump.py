from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import h5py
import numpy as np

conf = (SparkConf()).setAppName("testHDF5Conversion")
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

oceanTempsParquetFile = "/global/cscratch1/sd/gittens/CFSROparquet/finalmat.parquet"
hdf5file = "/global/cscratch1/sd/gittens/hdf5-export/data/oceanTemps-SMALL.hdf5"

fin = h5py.File(hdf5file, "r")
temps = fin.get("temperatures")
indices = np.arange(4000).tolist()

h5rows = np.array(temps[indices, :])

df = sqlContext.read.load(oceanTempsParquetFile)
sparkrows = np.array(map(lambda pair: pair[1], 
                         sorted(df.rdd.filter(lambda row : row[0] in indices)
                                  .collect(), key = lambda pair: pair[0])))

print np.linalg.norm(h5rows - sparkrows)

