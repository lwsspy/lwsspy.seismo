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

.. autofunction:: lwsspy.maps.plot_map.plot_topography

.. autofunction:: lwsspy.maps.plot_topography.plot_topography

.. figure:: figures/topography_europe.svg

    Shows the topography of Europe using Etopo1

.. figure:: figures/topography_europe.svg

    Shows the topography of the Earth using Etopo1. Note: Ice Sheets
    have not been implemented yet.



Math
++++

.. autofunction:: lwsspy.math.convm.convm

.. autofunction:: lwsspy.math.eigsort.eigsort

.. autofunction:: lwsspy.math.magnitude.magnitude

.. autoclass:: lwsspy.math.SphericalNN


Plotting Utilities
++++++++++++++++++

.. autofunction:: lwsspy.plot_util.figcolorbar.figcolorbar

.. autofunction:: lwsspy.plot_util.remove_ticklabels.remove_xticklabels

.. autofunction:: lwsspy.plot_util.remove_ticklabels.remove_yticklabels

.. autofunction:: lwsspy.plot_util.updaterc.updaterc


Seismology
++++++++++

.. autofunction:: lwsspy.seismo.cmt2inv.cmt2inv

.. autofunction:: lwsspy.seismo.cmt2stationxml.cmt2stationxml

.. autofunction:: lwsspy.seismo.cmtdir2stationxmldir.cmtdir2stationxmldir

.. autofunction:: lwsspy.seismo.inv2stationxml.inv2stationxml

.. autofunction:: lwsspy.seismo.perturb_cmt.perturb_cmt

.. autoclass:: lwsspy.seismo.source.CMTSource

.. autofunction:: lwsspy.seismo.validate_cmt.validate_cmt


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

.. autofunction:: lwsspy.seismo.specfem.readParfile.readParfile

.. autofunction:: lwsspy.seismo.specfem.stationxml2STATIONS.stationxml2STATIONS

.. autofunction:: lwsspy.seismo.specfem.stationxmldir2STATIONSdir.stationxmldir2STATIONSdir


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


Weather
+++++++

.. autofunction:: lwsspy.weather.drop2pickle.drop2pickle

.. autoclass:: lwsspy.weather.requestweather.requestweather

.. autoclass:: lwsspy.weather.weather.weather
