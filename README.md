# Collection of my codes.

Here I'm going to collect all codes hat I create. I will edit them etc.
`lwsspy` is the complement to `lwss`, which is for matlab codes.

## Installation

Hopefully, it works using `conda install lwsspy`.

## `PYTHONSTARTUP`

This repo contains a `startup.py` file that can be called when loading the 
python shell. If following line

```bash
export PYTHONSTARTUP=path/to/repo/startup.py
```

is added to tthe `~/.bashrc` file, Python will use the environment variable 
to load up the script. The script right now is set to load all of `pyplot`'s and
`numpy`'s functions without prefix as well as all of `lwsspy`'s functions.

This makes it possible to simply do small commands in Matlab style such as
`help(fakerelation)` or `plot(x,y,'o')`, etc.
