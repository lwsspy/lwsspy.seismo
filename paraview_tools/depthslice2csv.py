from paraview.simple import XMLUnstructuredGridReader
from paraview.simple import _DisableFirstRenderCameraReset
from paraview.simple import Slice
from paraview.simple import CreateView
from paraview.simple import SetActiveView
from paraview.simple import GetLayoutByName
from paraview.simple import Show
from paraview.simple import AssignViewToLayout
from paraview.simple import ExportView


def depthslice2csv(infile, outfile, depth):
    """Takes in a Spherical Earth Mesh and gets out a depth slice.

    Parameters
    ----------
    infile : str
        Mesh to slice
    outfile : str
        filename witthout ending to write the slice values to ``.csv`` file
    depth: float
        Depth to slice the spherical mesh from in KM.

    Returns
    -------
    None

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.09.17 19.00

    """

    # Earth parameters
    R_EARTH = 6371000.0/1000.0
    radius = (R_EARTH - depth)/R_EARTH

    # disable automatic camera reset on 'Show'
    _DisableFirstRenderCameraReset()

    # create a new 'XML Unstructured Grid Reader'
    rho = XMLUnstructuredGridReader(FileName=[infile])
    rho.PointArrayStatus = ['rho']

    # create a new 'Slice'
    slice1 = Slice(Input=rho)

    # Properties modified on slice1
    slice1.SliceType = 'Sphere'

    # Properties modified on slice1.SliceType
    slice1.SliceType.Center = [0.0, 0.0, 0.0]
    slice1.SliceType.Radius = radius

    # set active view
    SetActiveView(None)

    # Create a new 'SpreadSheet View'
    spreadSheetView1 = CreateView('SpreadSheetView')
    spreadSheetView1.ColumnToSort = ''
    spreadSheetView1.BlockSize = 1024
    # uncomment following to set a specific view size
    # spreadSheetView1.ViewSize = [400, 400]

    # show data in view
    Show(slice1, spreadSheetView1, 'SpreadSheetRepresentation')

    # get layout
    layout2 = GetLayoutByName("Layout #2")

    # assign view to a particular cell in the layout
    AssignViewToLayout(view=spreadSheetView1, layout=layout2, hint=0)

    # export view
    ExportView(outfile, view=spreadSheetView1)

    """
    BUILD IN CHECK THAT OPENS THE CSVFILE and checks whether it's empty
    as a sanity check
    """


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='infile',
                        help='Mesh to slice from',
                        required=True, type=str)
    parser.add_argument('-o', dest='outfile',
                        help='Output filename without file ending to save CSV.',
                        required=True, type=str)
    parser.add_argument('-d', dest='depth',
                        help='Depth in kilometers [km]',
                        required=True, type=float)
    args = parser.parse_args()

    # Run
    depthslice2csv(args.infile, args.outfile, args.depth)
