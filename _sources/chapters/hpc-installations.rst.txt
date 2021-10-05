High-Performance-Computing Installations
----------------------------------------

On thist page, I'm documenting the different ways of installing packages for
clusters. Specifically the ones that need ``MPI`` support, such as
`h5py <https://docs.h5py.org/en/stable/>`_, 
`tables <https://www.pytables.org/index.html>`_, etc.

.. warning::
    The most important thing for this whole page is that you use the **SAME
    MPI COMPILER** for everything.
    If one of the installations do not work with the MPI compiler of your choice
    chances are that they will not work in unison with your other installations,
    if at all.

    For most clusters, you can make an mpi compiler available through
    ``module load <your_favorite_mpi_compiler>`` 
    (I generally use ``openmpi/gcc``, which is not ideal, 
    because cluster specific compilers usually give you better support/speed).


HDF5 in Parallel with Fortran support
+++++++++++++++++++++++++++++++++++++

Below pasted a script that installs ``HDF5`` in Parallel without fuss.

.. code:: bash

    ### Install parallel HDF5
    # Set variables
    module load openmpi/gcc  # or compiler of your choice.
    MPICC=$(which mpicc)
    MPIF90=$(which mpif90)

    # HDF5
    HDF5_DESTDIR=<Path/to/where/you/to/install/hdf5>
    HDF5_LINK="https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.gz"
    HDF5_DIR="${HDF5_DIR}/build"
        
    # Get installation files
    wget -O hdf5.tar.gz $HDF5_LINK
	tar -xzvf hdf5.tar.gz --strip-components=1 -C $HDF5_DESTDIR 

    # Change directory to adios directory
    cd $HDF5_DESTIR

    # Check whether a previous build exists.
    if [ -d build ]; then
    	rm -rf build
    fi

    # Make build directory
    mkdir build

    # Configuration
    ./configure --enable-shared --enable-parallel \
        --enable-fortran --enable-fortran2003 \
        --prefix=$HDF5_DIR CC=$MPICC FC=$MPIF90

    # Make, check, and install
    make -j
    # make -j check  # not really necessary all the time
    make -j install


mpi4py
++++++

The installation of `mpi4py <https://mpi4py.readthedocs.io/en/stable/install.html>`_, 
is rather simple compared to the rest since it can be done directly from 
the command line.

.. code: bash

    module load openmpi/gcc
    MPICC=$(which mpicc)
    pip install mpi4py


h5py
++++

The newest versions of `h5py <https://docs.h5py.org/en/stable/>`_ is also
installed via ``pip`` when compiled against ``Parallel HDF5``.
This was not always the case.

So, given you have followed the script above and still have your 
``HDF5_DESTDIR`` environment variable loaded. ``h5py`` be installed using:

.. code:: bash

    # You need to have HDF5_DIR defined, check: echo $HDF5_DIR
    export CC=$(which mpicc)
    export HDF5_MPI="ON"
    pip install --no-binary=h5py h5py



PyTables
++++++++

`PyTables <https://www.pytables.org/index.html>`_ is a great tool to
interact (write and read) with Pandas DataFrames. However, it can be finicky to 
install with your already parallel ``HDF5`` and ``h5py`` packages installed.

You will have to set two, three environment variables to make the installation
possible.

.. code:: bash

    # Set hdf5 library and mpi compiler paths
    export HDF5_DIR=/path/to/your/parallel_hdf5_installation
    export CC=$(which mpicc)

    pip install tables


