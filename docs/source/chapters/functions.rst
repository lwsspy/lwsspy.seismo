Aaaalll the functions!
======================

Inversion
+++++++++

.. autoclass:: lwsspy.inversion.optimizer.Optimization
    :members: __init__

.. figure:: figures/optimization.svg

    Simple test of optimizing the Rosenbrock function.

.. figure:: figures/optimization_x4.svg

    Simple test of optimizing a 1D function. Note that we have 2 local
    minima which of course is a problem for local minimization schemes.


Maps
++++

.. autofunction:: lwsspy.maps.fix_map_extent.fix_map_extent

.. autofunction:: lwsspy.maps.plot_litho.plot_litho

.. autofunction:: lwsspy.maps.plot_map.plot_map

.. autofunction:: lwsspy.maps.plot_topography.plot_topography

.. figure:: figures/topography_europe.svg

    Shows the topography of Europe using Etopo1.

.. figure:: figures/topography_earth.svg

    Shows the topography of the Earth using Etopo1. Note: Ice Sheets
    have not been implemented yet.

.. autofunction:: lwsspy.maps.read_etopo.read_etopo

.. autofunction:: lwsspy.maps.read_litho.read_litho

.. autofunction:: lwsspy.maps.topocolormap.topocolormap



Math
++++

Coordinate Transformations
--------------------------

.. autofunction:: lwsspy.math.cart2geo.cart2geo

.. autofunction:: lwsspy.math.cart2pol.cart2pol

.. autofunction:: lwsspy.math.cart2sph.cart2sph

.. autofunction:: lwsspy.math.geo2cart.geo2cart

.. autofunction:: lwsspy.math.pol2cart.pol2cart

.. autofunction:: lwsspy.math.project2D.project2D

.. autofunction:: lwsspy.math.rotation_matrix.rotation_matrix

.. autofunction:: lwsspy.math.sph2cart.sph2cart

.. autofunction:: lwsspy.math.rodrigues.rodrigues


Miscellaneous
-------------

.. autofunction:: lwsspy.math.convm.convm

.. autofunction:: lwsspy.math.eigsort.eigsort

.. autofunction:: lwsspy.math.magnitude.magnitude

.. autoclass:: lwsspy.math.SphericalNN.SphericalNN
    :members:

Plotting Utilities
++++++++++++++++++

.. autofunction:: lwsspy.plot_util.figcolorbar.figcolorbar

.. autoclass:: lwsspy.plot_util.fixedpointcolornorm.FixedPointColorNorm

.. autofunction:: lwsspy.plot_util.nice_colorbar.nice_colorbar

.. autofunction:: lwsspy.plot_util.pick_data_from_image.pick_data_from_image

.. autofunction:: lwsspy.plot_util.plot_label.plot_label

.. autofunction:: lwsspy.plot_util.remove_ticklabels.remove_xticklabels

.. autofunction:: lwsspy.plot_util.remove_ticklabels.remove_yticklabels

.. autofunction:: lwsspy.plot_util.updaterc.updaterc

.. autofunction:: lwsspy.plot_util.view_colormap.view_colormap


Seismology
++++++++++

.. autofunction:: lwsspy.seismo.cmt2inv.cmt2inv

.. autofunction:: lwsspy.seismo.cmt2stationxml.cmt2stationxml

.. autofunction:: lwsspy.seismo.cmtdir2stationxmldir.cmtdir2stationxmldir

.. autofunction:: lwsspy.seismo.inv2stationxml.inv2stationxml

.. autofunction:: lwsspy.seismo.perturb_cmt.perturb_cmt

.. autoclass:: lwsspy.seismo.source.CMTSource

.. autofunction:: lwsspy.seismo.validate_cmt.validate_cmt


Signal Processing
-----------------

.. autofunction:: lwsspy.seismo.process.compare_trace.least_square_error

.. autofunction:: lwsspy.seismo.process.compare_trace.cross_correlation

.. autofunction:: lwsspy.seismo.process.compare_trace.trace_length

.. automodule:: lwsspy.seismo.process.process
    :members:

.. autofunction:: lwsspy.seismo.process.process_wrapper.process_wrapper

.. automodule:: lwsspy.seismo.process.rotate
    :members:

.. automodule:: lwsspy.seismo.process.rotate_utils
    :members:




SPECFEM handling
----------------

.. autofunction:: lwsspy.seismo.specfem.cmt2rundir.cmt2rundir

.. autofunction:: lwsspy.seismo.specfem.cmt2simdir.cmt2simdir

.. autofunction:: lwsspy.seismo.specfem.cmt2STATIONS.cmt2STATIONS

.. autofunction:: lwsspy.seismo.specfem.cmtdir2rundirs.cmtdir2rundirs

.. autofunction:: lwsspy.seismo.specfem.cmtdir2simdirs.cmtdir2simdirs

.. autofunction:: lwsspy.seismo.specfem.createsimdir.createsimdir

.. autofunction:: lwsspy.seismo.specfem.getsimdirSTATIONS.getsimdirSTATIONS

.. autofunction:: lwsspy.seismo.specfem.inv2STATIONS.inv2STATIONS

.. autofunction:: lwsspy.seismo.specfem.plot_csv_depth_slice.plot_csv_depth_slice

.. autofunction:: lwsspy.seismo.specfem.plot_specfem_xsec_depth.plot_specfem_xsec_depth

.. autofunction:: lwsspy.seismo.specfem.read_parfile.read_parfile

.. autofunction:: lwsspy.seismo.specfem.read_specfem_xsec_depth.read_specfem_xsec_depth

.. autofunction:: lwsspy.seismo.specfem.stationxml2STATIONS.stationxml2STATIONS

.. autofunction:: lwsspy.seismo.specfem.stationxmldir2STATIONSdir.stationxmldir2STATIONSdir

.. autofunction:: lwsspy.seismo.specfem.write_parfile.write_parfile


Statistics
++++++++++

.. autofunction:: lwsspy.statistics.fakerelation.fakerelation

.. figure:: figures/modelled_covarying_dataset.svg

    Modeled covarying data sets.

.. autofunction:: lwsspy.statistics.errorellipse.errorellipse

.. figure:: figures/error_ellipse.svg

    Generating the error ellipse for a covarying dataset.

.. autofunction:: lwsspy.statistics.gaussian2d.gaussian2d

.. figure:: figures/gaussian2d.svg

    Forward modelling a 2D Gaussian distribution.

.. autofunction:: lwsspy.statistics.fitgaussian2d.fitgaussian2d

.. figure:: figures/fitgaussian2d.svg

.. autofunction:: lwsspy.statistics.distlist.distlist

.. figure:: figures/distlist.svg

    Creating a list of distributions.

.. autofunction:: lwsspy.statistics.clm.clm

.. figure:: figures/clm.svg

    Showing that the central limit theorem holds. Note that the convolution
    limits are not actually correct, and this plot is solely for illustration.

Utilities
+++++++++

I/O
---

.. automodule:: lwsspy.utils.io
    :members:

Print Utilities
---------------

.. automodule:: lwsspy.utils.output
    :members:

Miscellaneous
-------------

.. autofunction:: lwsspy.utils.chunks.chunks

.. autofunction:: lwsspy.utils.cpu_count.cpu_count

.. autofunction:: lwsspy.utils.pixels2data.pixels2data

.. automodule:: lwsspy.utils.threadwork
    :members:

Weather
+++++++

.. autofunction:: lwsspy.weather.drop2pickle.drop2pickle

.. autoclass:: lwsspy.weather.requestweather.requestweather

.. autoclass:: lwsspy.weather.weather.weather
