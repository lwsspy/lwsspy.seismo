#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is mostly taken from Wenjie Lei (2020) to create CMTSOLUTION file sets
for  forward simulations.
"""

# External imports
import os
import sys
import glob
import yaml
from copy import deepcopy
from typing import Union

# Internal imports
from .source import CMTSource
from .validate_cmt import validate_cmt


def perturb_one_var(origin_cmt: CMTSource, pert_type: str, 
                    pert_value: float, outputdir: str):
    """This file takes in a CMTSource and a perturbation type and value to
    and perturbs the CMTSource with respect to that one value.

    Args:
        origin_cmt (CMTSource):
            CMTSource
        pert_type (str): 
            perturbation type: 
            ``['m_rr', 'm_tt', 'm_pp', 'm_rt', 'm_rp', 'm_tp', 
            'depth_in_m', 'latitude', 'longitude']``
        pert_value (float):
            magnitude of perturbation
        outputdir (str):
            Output directory/where to write the perturbed file with a suffix 
            depending on the perturbation.
    
    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)

    """
    cmt = deepcopy(origin_cmt)
    if pert_type in ['m_rr', 'm_tt', 'm_pp', 'm_rt', 'm_rp', 'm_tp']:
        # if perturb moment tensor, set all moment tensor to zero
        cmt.m_rr = 0.0
        cmt.m_tt = 0.0
        cmt.m_pp = 0.0
        cmt.m_rt = 0.0
        cmt.m_rp = 0.0
        cmt.m_tp = 0.0
        setattr(cmt, pert_type, pert_value)
    elif pert_type in ['depth_in_m', 'latitude', 'longitude']:
        attr = getattr(cmt, pert_type)
        setattr(cmt, pert_type, attr+pert_value)

    # Check whether perturbation is ok!
    validate_cmt(cmt)

    # Depending on perturbation choose CMTSOLUTION suffix
    suffix_dict = {'m_rr': "Mrr", 'm_tt': "Mtt", 'm_pp': "Mpp", 'm_rt': "Mrt",
                   'm_rp': "Mrp", 'm_tp': "Mtp", "depth_in_m": "dep",
                   "latitude": "lat", "longitude": "lon"}
    suffix = suffix_dict[pert_type]
    outputfn = os.path.join(outputdir, "%s_%s" % (cmt.eventname, suffix))

    # Write file
    cmt.write_CMTSOLUTION_file(outputfn)


def perturb_cmt(input_cmtfile: str, output_dir: str = ".",
                dmoment_tensor: Union[float, None] = None,
                dlongitude: Union[float, None] = None,
                dlatitude: Union[float, None] = None,
                ddepth_km: Union[float, None] = None):
    """Perturb a single CMTSOLUTION given certain perturbation value.
    If the perturbation value is ``None``. The file will not be perturbed with 
    respect to that value.

    Args:
        input_cmtfile (str):
            Path to CMTSOLUTION file
        output_dir (str, optional):
            Output directory, where to save the perturbed file.
            Defaults to ".".
        dmoment_tensor (Union[float, None], optional):
            Perturbation in seismic moment. Defaults to None.
        dlongitude (Union[float, None], optional):
            Perturbation in longitude. Defaults to None.
        dlatitude (Union[float, None], optional):
            Perturbation in latitude. Defaults to None.
        ddepth_km (Union[float, None], optional):
            Perturbation in depth. Defaults to None.

    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)

    """
    # Create output directory if it doesn't exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create CMTSource
    cmt = CMTSource.from_CMTSOLUTION_file(input_cmtfile)

    if dmoment_tensor is not None:
        pert_type_list = ['m_rr', 'm_tt', 'm_pp', 'm_rt', 'm_rp', 'm_tp']
        for pert_type in pert_type_list:
            perturb_one_var(cmt, pert_type, dmoment_tensor, output_dir)

    if dlongitude is not None:
        perturb_one_var(cmt, "longitude", dlongitude, output_dir)

    if dlatitude is not None:
        perturb_one_var(cmt, "latitude", dlatitude, output_dir)

    if ddepth_km is not None:
        perturb_one_var(cmt, "depth_in_m", ddepth_km*1000.0, output_dir)


def perturb_cmt_dir(cmtdir: str = "./CMT", outputdir: str = "./",
                    dmoment_tensor: Union[float, None] = None,
                    dlongitude: Union[float, None] = None,
                    dlatitude: Union[float, None] = None,
                    ddepth_km: Union[float, None] = None):
    """[summary]

    Args:
        cmtdir (str, optional): 
            Directory with original CMTs. Defaults to "./CMT".
        outputdir (str, optional):
            Directory with. Defaults to "./CMT.perturb".
        dmoment_tensor (Union[float, None], optional):
            Perturbation in seismic moment. Defaults to None.
        dlongitude (Union[float, None], optional):
            Perturbation in longitude. Defaults to None.
        dlatitude (Union[float, None], optional):
            Perturbation in latitude. Defaults to None.
        ddepth_km (Union[float, None], optional):
            Perturbation in depth. Defaults to None.

    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)
    """
    # Get all CMT files
    cmtfiles = glob.glob(os.path.join(cmtdir, "*"))
    print("Number of CMT files: %d" % len(cmtfiles))

    for i, _file in enumerate(cmtfiles):
        print(f"#{i+1:0>5}/{len(cmtfiles)}:{_file:_>50}")
        perturb_cmt(_file, output_dir=outputdir,
                    dmoment_tensor=dmoment_tensor, ddepth_km=ddepth_km,
                    dlatitude=dlatitude, dlongitude=dlongitude)


if __name__ == '__main__':

    # Perturbation
    dmoment_tensor = 1.0e23
    ddepth_km = 2.0
    dlatitude = 0.02
    dlongitude = 0.02

    # Directories
    cmtdir = "./CMT"
    outputdir = "./CMT.perturb"

    # Perturb directory of CMTs
    perturb_cmt_dir(cmtdir=cmtdir, outputdir=outputdir,
                    dmoment_tensor=dmoment_tensor, ddepth_km=ddepth_km,
                    dlatitude=dlatitude, dlongitude=dlongitude)
    