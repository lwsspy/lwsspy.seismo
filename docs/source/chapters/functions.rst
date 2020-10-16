Aaaalll the functions!
======================

Inversion
+++++++++

.. autoclass:: lwsspy.inversion.optimization.Optimizer

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

.. image:: figures/modelled_covarying_dataset.svg

.. autofunction:: lwsspy.statistics.errorellipse.errorellipse

.. image:: figures/error_ellipse.svg

.. autofunction:: lwsspy.statistics.gaussian2d.gaussian2d

.. image:: figures/gaussian2d.svg

.. autofunction:: lwsspy.statistics.fitgaussian2d.fitgaussian2d

.. image:: figures/fitgaussian2d.svg

.. autofunction:: lwsspy.statistics.distlist.distlist

.. image:: figures/distlist.svg

.. autofunction:: lwsspy.statistics.clm.clm

.. image:: figures/clm.svg

Weather
+++++++

.. autofunction:: lwsspy.weather.drop2pickle.drop2pickle

.. autoclass:: lwsspy.weather.requestweather.requestweather

.. autoclass:: lwsspy.weather.weather.weather
