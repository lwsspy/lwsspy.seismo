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