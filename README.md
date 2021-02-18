# Collection of my codes.

Here I'm going to collect all codes hat I create. I will edit them etc.
`lwsspy` is the complement to `lwss`, which is for matlab codes.

---

## Installation

Hopefully, it works using `conda install lwsspy` will work in the future.

### Installation using Pip

However, 

```bash
git clone git@github.com:lsawade/lwsspy.git
cd lwsspy
pip install .
```

Should work. Use `pip install -e .` for development mode.

### Summit/Traverse/Parallel

It's a bit more convoluted since we will have to install some things using,
`conda`, then install `parallel HDF5`, `mpi4py`, and `h5py`. And then,
we can install the rest of the functions via `pip`.


1. `conda env create -f summitenv.yml`
2. Check the documentation for the installation of Parallel HDF5 on the cluster, and
subsequent installations of `mpi4py`, and `h5py`.
3. `pip install summitreq.yml`


---

## `PYTHONSTARTUP`

This repo contains a `startup.py` file that can be called when loading the 
python shell. If following line

```bash
export PYTHONSTARTUP=path/to/repo/startupfiles/python.py
```

is added to the `~/.bashrc` file, Python will use the environment variable 
to load up the script. The script right now is set to load all of `pyplot`'s and
`numpy`'s functions without prefix as well as all of `lwsspy`'s functions.

This makes it possible to simply do small commands in Matlab style such as
`help(fakerelation)` or `plot(x,y,'o')`, etc.

## Autoreload modules before execution

In addition to the Python startup file. Ipython has the ability to reload
modified modules on the fly. This is extremely convenient:

Simply run the line:

```bash
cp path/to/repo/startupfiles/ipython.ipy ~/.ipython/profile_default/startup/
```

To run the lines in ipython.ipy. The lines are the following:

```
# Activate autoreload
%load_ext autoreload
%autoreload 2
```
