NumpyDoc Tips
+++++++++++++

.. warning::

    This may look long and tedious, but IDE's such as VSCode can autogenerate
    a large chunk of the ``docstring`` using the functions inputs, outputs, and
    raise errors. That way you only have to fill out descriptions of your 
    functions and extra things that you want the user to know.

Documenting functions
---------------------

Functions have a set of expected headers, which all are optional, but should be 
thought about when writing the ``docstring`` for ``numpydoc``. I include a sample
docstring below to show how a normal funnction should be documented. 
Note that the function's docstring is not the same as in the package, since 
that would be too long for tthe sake of the example.

.. code:: python

    def pick_data_from_image(infile: str, outfile: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function's purpose is the replication of previously published data.
        
        ``csv`` output format:

        ::
            # Created from following file:
            x0, y0
            x1, y1
            ...

        Parameters
        ----------
        infile: str
            Image file name to be loaded by the function. The image should be
            cropped to the axes edges.
        outfile: Union[str, None], optional
            output file name. Must contain ``.csv`` file ending

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            x, y vectors of the picked data.
        
        Raises
        ------
        ValueError
            if file ending is not ``.csv``

        See Also
        --------
        lwsspy.utils.pixels2data.pixels2data : Scaling from pixels to dataunits


        Examples
        --------

        >>> import lwsspy as lpy
        >>> x,y = lpy.pick_data_from_image('test.png', 'testdata.csv')

        Notes
        -----

        .. note::

            Some note

        .. warning::

            some  warning

        :Authors:
            Lucas Sawade (lsawade@princeton.edu)

        :Last Modified:
            2020.01.06 11.00


        """

        # Just pseudo-code...
        file = open(infile)
        xy = pickdata(file)
        savefile(outfile, xy)
        return x,y
        

Usage:

- ``Parameters`` section to explain the inputs.
- ``Returns`` section for outputs
- ``Raises`` section for errors that are raised by the function
- ``See also`` section to link other functions to the function you want
  to document.
- ``Examples`` section to describe the funcitonality of your code
- ``Notes`` section for notes and possible author info


Documenting Classes
-------------------

Documenting class is very similar to documenting functions, but there are
some extra directives you want for the ``class`` description, which are 
``Methods`` and ``Attributes``. The method section is very optional and should
only be used if not all methods are supposed to be documented.

Example:

.. code:: python

    class SphericalNN(object):
        """Spherical nearest neighbour queries using scipy's fast kd-tree
        implementation.

        Attributes
        ----------
        data : numpy.ndarray
            cartesian point data array [x,y,z]
        kd_tree : scipy.spatial.cKDTree
            a KDTree used to query data


        Methods
        -------
        query(qlat, qlon)
            Query a set of latitudes and longitudes
        SphericalNN.query_pairs(maximum_distance)
            Find pairs of points that are within a certain distance of each other
        SphericalNN.interp(data, qlat, qlon)
            Use the kdtree to interpolate data corresponding 
            to the points of the Kdtree onto a new set of points using nearest 
            neighbor interpolation or weighted nearest neighbor 
            interpolation (default).

        Notes
        -----

        :Authors:
            Lucas Sawade (lsawade@princeton.edu)

        :Last Modified:
            2020.01.06 14.00

        """

        def __init__(self, lat, lon):
            """Initialize class

            Parameters
            ----------
            lat : numpy.ndarray
                latitudes
            lon : numpy.ndarray
                longitudes
            """
            cart_data = self.spherical2cartesian(lat, lon)
            self.data = cart_data
            self.kd_tree = cKDTree(data=cart_data, leafsize=10)

        def othermethodsasmetioned():
            pass

In the class description we describe two extra section as mention earlier. 
That's it. Method docstrings should be the same as function docstrings, but make
sure you omit the ``self`` statement.

            
