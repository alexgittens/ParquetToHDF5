dumpToHDF5Chunks.py loads a specified IndexedRowMatrix in Spark from Parquet, without caching, and writes each partition out as an HDF5 file (hence it makes sense to ensure there are fewer partitions) containing the actual rows in that partition as one dataset, and the corresponding row indices as another dataset. We don't assume that the rows are stored in consecutive order.
concat.py uses MPI4Py to merge these HDF5 files into one big file where the rows are in consecutive order. It does this by dividing the HDF5 files evenly across all the processes, then the root process asks each node to return all the rows it contains within a certain sized window that moves along the rows of the final matrix. As each chunk of rows is received from the processes, the root sorts them into consecutive order and writes them out to the final file.
runconcat.slrm is a SLURM batch file for use on Cori for running concat.py
mpiless-concat.py is for error-checking: it merges the HDF5 files by opening each one sequentially and writing its rows directly to the correct row in the final matrix.

TODO:
 - give a script front-end to dumpToHDF5Chunks.py and concat.py to let them be called sequentially on an arbitrary dataset and potentially also do other operations like transposition at the same time
 - do better error-checking, maybe write formal tests

