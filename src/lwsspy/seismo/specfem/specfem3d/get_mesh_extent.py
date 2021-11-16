import os
import numpy as np


def get_mesh_extent(specfemdir):

    outputmesher = os.path.join(
        specfemdir, 'OUTPUT_FILES', "output_generate_databases.txt")

    with open(outputmesher, "r") as f:

        lines = f.readlines()

    for line in lines:
        if "Xmin and Xmax" in line:
            xlims = np.array(line.split()[-2:], dtype=float)

        if "Ymin and Ymax" in line:
            ylims = np.array(line.split()[-2:], dtype=float)

        if "Zmin and Zmax" in line:
            zlims = np.array(line.split()[-2:], dtype=float)

    return xlims, ylims, zlims
