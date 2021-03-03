Seismology
----------

Just a set of function and scripts, I have created for figures that I like,
as well as other notes I deem important in terms of coding in seismology.


.. _station-map:

Plotting a good looking station map
+++++++++++++++++++++++++++++++++++

I have created a script that is working well for up to 10 networks with the 
default settings, which use matplotlib's ``tab10`` qualatative colormap.
For more networks, the :func:`lwsspy.plot_inventory` function takes in 
a colormap option which takes `x` colors for `x` networks sequentially from
a matplotlib colormap. A full script for writing the figure is shown below.

.. literalinclude:: figures/scripts/station_map.py
  :language: python

With the output:

.. image:: figures/station_map.svg


.. _ideal-stf:

Ideal Source Time Functions
++++++++++++++++++++++++++++++++

Just to be on the safe side and to not have to redo it all the time, I created
some functions that compute sourcetime functions with certain dominant
frequencies. 

**A Gaussian pulse**:

.. math::
    
    s(t) = e^{-{\left(4f_0  (t-t_0)\right)}^2}

**and its derivative**:

.. math::
    
    s(t) = -(8f_0) (t-t_0) e^{-{\left(4f_0  (t-t_0)\right)}^2}

A script to plot the functions and their spectra below:

.. literalinclude:: figures/scripts/gaussians.py
  :language: python

With the output:

.. image:: figures/gaussians.svg

The functions are documented here :py:func:`lwsspy.seismo.gaussiant.gaussiant` 
and :py:func:`lwsspy.seismo.gaussiant.dgaussiant` 

