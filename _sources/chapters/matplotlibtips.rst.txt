Tips And Tricks for Matplotlib
==============================

This page is not really a part of the documentation of the package, but rather
some tips and tricks I have gathered over time.


.. _partial-rasterize:

Partially rasterizing your PDF output
+++++++++++++++++++++++++++++++++++++

You probably have tried outputting your meshplot in matplotlib and wondered
why the heck it is taking soo long! 
The explanation is simple. Every coordinate and data combination is creating a 
box (with coordinattes) with a color that has to be output and written to 
the PDF file.
When your figure includes multiple meshplots with 1000x1000 plots, 
the number of boxes becomes very large and the number of colors/coordinates even
larger.
One way to get around it, is simply saving it as `png` file.
That however doesn't make your plots exactly publishable.

A workaround is partially rasterizing your plots. That can be done with the
following command:

.. literalinclude:: figures/scripts/rasterize.py
  :language: python


It works for both `.svg` and `.pdf`. Probably others too, but I haven't tried.

Below the image produced by the code above

.. image:: figures/test_rasterize.svg



Make x/y labels invisible on shared axes plots
++++++++++++++++++++++++++++++++++++++++++++++

To make plots with subfigures more beautiful, you may want to remove axes
labels if the plots share the axes!

.. literalinclude:: figures/scripts/remove_labels.py
  :language: python

.. image:: figures/remove_labels.svg


Make the figure background transparent when exporting
+++++++++++++++++++++++++++++++++++++++++++++++++++++

``matplotlib.pyplot.savefig`` has a keyword ``transparent`` which makes the 
background of figures when exported to formats, such as ``png`` or ``svg``,
transparent. Simply save your figure using

.. code:: python

    import matplotlib.pyplot as plt
    
    # your figure code goes here
    
    plt.savefig(<output_filename>, transparent=True)


See :ref:`station-map` for a use of the figure. Note that the rtd-themes 
background is not white, but off white, if the figure background would be 
visible, you would see it. For a converse example, see :ref:`partial-rasterize`.


Multiple Locator for the Axes
+++++++++++++++++++++++++++++

Sometimes it is convenient to show things as multiples of a certain values. 
A good example are radians, often times cyclical motions are easier 
illustrated as multiples of pi instead of it's value in radians. 
Using functions from ``lwsspy`` this can be done two ways, where the second 
one is arguably more elegant.

.. literalinclude:: figures/scripts/multiple_locator.py
  :language: python

.. image:: figures/multiple_locator.svg


Plotting a Line with Variable Color and Width
+++++++++++++++++++++++++++++++++++++++++++++

Below shown the usage of a convenience function included in ``lwsspy`` which 
makes it easy to plot lines with variable color and width. It takes
in ``x``, ``y``, and ``z`` and parses all other ``*args`` and ``**kwargs`` to 
a ``LineCollection`` and returns the ``LineCollection`` as well as a 
``ScalarMappable`` to enable simple colorbar creation.

.. literalinclude:: figures/scripts/plot_xyz_line.py
  :language: python

.. image:: figures/xyz_line.svg


An example using the ``lwsspy.plot_xyz_line`` of ancient reef topography 
in Nevada.

.. image:: figures/nevada_dem.png


rotational view.


.. raw:: html

    <video width="100%" height="auto" controls src="../_static/hello.mp4" type="video/mp4" autoplay loop></video>