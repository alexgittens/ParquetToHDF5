dumpToHDF5Chunks.py loads a specified IndexedRowMatrix in Spark from Parquet,
without caching, and writes each partition out as an HDF5 file (hence it makes
sense to ensure there are fewer partitions) containing the actual rows in that
partition as one dataset, and the corresponding row indices as another dataset.
We don't assume that the rows are stored in consecutive order.

concat.py uses MPI4Py to merge these HDF5 files into one big file where the
rows are in consecutive order. It does this by dividing the input HDF5 files evenly
across all the processes and using several processes to write to the same file simultaneously using hdf5-parallel.
A certain size window moves along the rows of the final matrix, and each process sends 
the relevant rows from all of its assigned files to the writers handling the appropriate part of that window.
As each chunk of rows is received from the processes, the
writers sort them into consecutive order and writes them out to the final file.

partition.py uses MPI4Py to divide the large HDF5 file into many smaller chunks
and writes them out to compressed HDF5 files for faster transfer, in an
embarassingly parallel manner.

TODO:
 - give a script front-end to dumpToHDF5Chunks.py, concat.py, partition.py to let them be called sequentially on an arbitrary dataset and potentially also do other operations like transposition at the same time
 - do better error-checking, maybe write formal tests

