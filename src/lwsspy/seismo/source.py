#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Source and Receiver classes of Instaseis.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    Martin van Driel (Martin@vanDriel.de), 2014
    Wenjie Lei (lei@princeton.edu), 2016
    Lucas Sawade (lsawade@princeton.edu), 2019

:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
import numpy as np
import typing as tp
from obspy import UTCDateTime, read_events
from obspy.imaging.beachball import beach
from obspy.core.event import Event
import warnings
from . import sourcedecomposition
from inspect import getmembers, isfunction
from ..plot.axes_from_axes import axes_from_axes
from ..plot.plot_label import plot_label
from ..plot.get_aspect import get_aspect
from ..plot.updaterc import updaterc
from ..plot.midpointcolornorm import MidpointNormalize
from .eqpar import eqpar
import matplotlib.pyplot as plt
from matplotlib import transforms


class CMTSource(object):
    """
    Class to handle a seismic moment tensor source including a source time
    function.
    """

    def __init__(
            self, origin_time: tp.Union[UTCDateTime, float] = UTCDateTime(0),
            pde_latitude=0.0, pde_longitude=0.0, mb=0.0, ms=0.0,
            pde_depth_in_m=0.0, region_tag='', eventname='',
            cmt_time: tp.Union[UTCDateTime, float] = UTCDateTime(0), 
            half_duration=0.0, latitude=0.0, longitude=0.0, depth_in_m=0.0, 
            m_rr=0.0, m_tt=0.0, m_pp=0.0, m_rt=0.0, m_rp=0.0, m_tp=0.0):
        """
        :param latitude: latitude of the source in degree
        :param longitude: longitude of the source in degree
        :param depth_in_m: source depth in m
        :param m_rr: moment tensor components in r, theta, phi in Nm
        :param m_tt: moment tensor components in r, theta, phi in Nm
        :param m_pp: moment tensor components in r, theta, phi in Nm
        :param m_rt: moment tensor components in r, theta, phi in Nm
        :param m_rp: moment tensor components in r, theta, phi in Nm
        :param m_tp: moment tensor components in r, theta, phi in Nm
        :param time_shift: correction of the origin time in seconds. only
            useful in the context of finite sources
        :param sliprate: normalized source time function (sliprate)
        :param dt: sampling of the source time function
        :param origin_time: The origin time of the source. This will be the
            time of the first sample in the final seismogram. Be careful to
            adjust it for any time shift or STF (de)convolution effects.
        """
        self.origin_time = origin_time
        self.pde_latitude = pde_latitude
        self.pde_longitude = pde_longitude
        self.pde_depth_in_m = pde_depth_in_m
        self.mb = mb
        self.ms = ms
        self.region_tag = region_tag
        self.eventname = eventname
        self.cmt_time = cmt_time
        self.half_duration = half_duration
        self.latitude = latitude
        self.longitude = longitude
        self.depth_in_m = depth_in_m
        self.m_rr = m_rr
        self.m_tt = m_tt
        self.m_pp = m_pp
        self.m_rt = m_rt
        self.m_rp = m_rp
        self.m_tp = m_tp

    @classmethod
    def from_CMTSOLUTION_file(cls, filename):
        """
        Initialize a source object from a CMTSOLUTION file.
        :param filename: path to the CMTSOLUTION file
        """

        with open(filename, "rt") as f:
            line = f.readline()
            origin_time = line[5:].strip().split()[:6]
            values = list(map(int, origin_time[:-1])) \
                + [float(origin_time[-1])]
            try:
                origin_time = UTCDateTime(*values)
            except (TypeError, ValueError):
                warnings.warn("Could not determine origin time from line: %s"
                              % line)
                origin_time = UTCDateTime(0)
            otherinfo = line[4:].strip().split()[6:]
            pde_lat = float(otherinfo[0])
            pde_lon = float(otherinfo[1])
            pde_depth_in_m = float(otherinfo[2]) * 1e3
            mb = float(otherinfo[3])
            ms = float(otherinfo[4])
            region_tag = ' '.join(otherinfo[5:])

            eventname = f.readline().strip().split()[-1]
            time_shift = float(f.readline().strip().split()[-1])
            cmt_time = origin_time + time_shift
            half_duration = float(f.readline().strip().split()[-1])
            latitude = float(f.readline().strip().split()[-1])
            longitude = float(f.readline().strip().split()[-1])
            depth_in_m = float(f.readline().strip().split()[-1]) * 1e3

            #                                                 unit: N/m
            m_rr = float(f.readline().strip().split()[-1])  # / 1e7
            m_tt = float(f.readline().strip().split()[-1])  # / 1e7
            m_pp = float(f.readline().strip().split()[-1])  # / 1e7
            m_rt = float(f.readline().strip().split()[-1])  # / 1e7
            m_rp = float(f.readline().strip().split()[-1])  # / 1e7
            m_tp = float(f.readline().strip().split()[-1])  # / 1e7

        return cls(origin_time=origin_time,
                   pde_latitude=pde_lat, pde_longitude=pde_lon, mb=mb, ms=ms,
                   pde_depth_in_m=pde_depth_in_m, region_tag=region_tag,
                   eventname=eventname, cmt_time=cmt_time,
                   half_duration=half_duration, latitude=latitude,
                   longitude=longitude, depth_in_m=depth_in_m,
                   m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_rt=m_rt,
                   m_rp=m_rp, m_tp=m_tp)

    @classmethod
    def from_quakeml_file(cls, filename: str):
        """
        Initialiaze a source object from a quakeml file
        :param filename: path to a quakeml file
        """
        cat = read_events(filename)
        event = cat[0]

        return cls.from_event(event)

    @classmethod
    def from_sdr(cls, s, d, r, M0=1.0, **kwargs):
        """definition from Stein and Wysession
        s is the strike (phi_f), d is the dip (delta), and r is the slip angle
        (lambda). IM ASSIGNING THE COMPONENTS WRONG. SEE EMAIL TO YURI!!!!!

        Note How in Stein and Wysession the z-axis points UP (Figure 4.2-2) and
        in Aki and Richards the z-axis points down (Figure 4.20). 
        This results in a sign changes for the fault normal and slip vector. 

        Convention for the code here is 
        X-North, Y-East, Z-Down
        R-Down, Theta-North, Phi-East


        """

        # To radians
        s = np.radians(s)
        d = np.radians(d)
        r = np.radians(r)
        

        # Fault normal
        n = np.array([
            - np.sin(d) * np.sin(s),
            + np.sin(d) * np.cos(s),
            - np.cos(d)
        ])

        # Slip vector
        d = np.array([
            np.cos(r) * np.cos(s) + np.cos(d) * np.sin(r) * np.sin(s),
            np.cos(r) * np.sin(s) - np.cos(d) * np.sin(r) * np.cos(s),
            - np.sin(r) * np.sin(d)
        ])

        # Compute moment tensor Stein and Wysession Style
        Mx = M0 * (np.outer(n, d) + np.outer(d, n))

        # # Full terms from AKI & Richards (Box 4.4)
        # Mxx = -M0 * (np.sin(d) * np.cos(r) * np.sin(2*s) + np.sin(2*d) * np.sin(r) * np.sin(s)**2)
        # Mxy = +M0 * (np.sin(d) * np.cos(r) * np.cos(2*s) + 0.5 * np.sin(2*d) * np.sin(r) * np.sin(2*s))
        # Mxz = -M0 * (np.cos(d) * np.cos(r) * np.cos(s) + np.cos(2*d) * np.sin(r) * np.sin(s))
        # Myy = +M0 * (np.sin(d) * np.cos(r) * np.sin(2*s) - np.sin(2*d) * np.sin(r) * np.cos(s)**2)
        # Myz = -M0 * (np.cos(d) * np.cos(r) * np.sin(s) - np.cos(2*d) * np.sin(r) * np.cos(s))
        # Mzz = +M0 * np.sin(2*d) * np.sin(r)
        
        # # Moment tensor creation
        # Mx = np.array( [
        #     [Mxx, Mxy, Mxz],
        #     [Mxy, Myy, Myz],
        #     [Mxz, Myz, Mzz]
        # ])

        Mr = np.array([
            [ Mx[2, 2],  Mx[2, 0], -Mx[2, 1]],
            [ Mx[0, 2],  Mx[0, 0], -Mx[0, 1]],
            [-Mx[1, 2], -Mx[1, 0],  Mx[1, 1]],
        ])

        return cls(m_rr=Mr[0, 0], m_tt=Mr[1, 1], m_pp=Mr[2, 2], m_rt=Mr[0, 1],
                   m_rp=Mr[0, 2], m_tp=Mr[1, 2])

    @classmethod
    def from_event(cls, event: Event):

        for origin in event.origins:
            if origin.origin_type == 'centroid':
                cmtsolution = origin
            else:
                pdesolution = origin

        origin_time = pdesolution.time
        pde_lat = pdesolution.latitude
        pde_lon = pdesolution.longitude
        pde_depth_in_m = pdesolution.depth
        mb = 0.0
        ms = 0.0
        for mag in event.magnitudes:
            if mag.magnitude_type == "Mb":
                mb = mag.mag
            elif mag.magnitude_type == "MS":
                ms = mag.mag
        # Get region tag
        try:
            region_tag = cmtsolution.region
        except Exception:
            try:
                region_tag = cmtsolution.region
            except Exception:
                warnings.warn("Region tag not found.")

        for descrip in event.event_descriptions:
            if descrip.type == "earthquake name":
                eventname = descrip.text
            else:
                eventname = ""
        cmt_time = cmtsolution.time
        focal_mechanism = event.focal_mechanisms[0]
        half_duration = \
            focal_mechanism.moment_tensor.source_time_function.duration/2.0
        latitude = cmtsolution.latitude
        longitude = cmtsolution.longitude
        depth_in_m = cmtsolution.depth
        tensor = focal_mechanism.moment_tensor.tensor
        m_rr = tensor.m_rr * 1e7
        m_tt = tensor.m_tt * 1e7
        m_pp = tensor.m_pp * 1e7
        m_rt = tensor.m_rt * 1e7
        m_rp = tensor.m_rp * 1e7
        m_tp = tensor.m_tp * 1e7

        return cls(origin_time=origin_time,
                   pde_latitude=pde_lat, pde_longitude=pde_lon, mb=mb, ms=ms,
                   pde_depth_in_m=pde_depth_in_m, region_tag=region_tag,
                   eventname=eventname, cmt_time=cmt_time,
                   half_duration=half_duration, latitude=latitude,
                   longitude=longitude, depth_in_m=depth_in_m,
                   m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_rt=m_rt,
                   m_rp=m_rp, m_tp=m_tp)

    @classmethod
    def from_dictionary(cls, d):
        """
        Initialize a source object from a CMTSOLUTION file.

        :param dictionary: dictionary
        """

        origin_time = UTCDateTime(d["origin_time"][3:])
        pde_lat = d["pde_latitude"]
        pde_lon = d["pde_longitude"]
        pde_depth_in_m = d["pde_depth_in_m"]
        mb = d["mb"]
        ms = d["ms"]
        region_tag = d["region_tag"]

        eventname = d["eventname"]
        cmt_time = UTCDateTime(d["cmt_time"][3:])
        half_duration = d["half_duration"]
        latitude = d["latitude"]
        longitude = d["longitude"]
        depth_in_m = d["depth_in_m"]

        # unit: N/m
        m_rr = d["m_rr"]
        m_tt = d["m_tt"]
        m_pp = d["m_pp"]
        m_rt = d["m_rt"]
        m_rp = d["m_rp"]
        m_tp = d["m_tp"]

        return cls(origin_time=origin_time,
                   pde_latitude=pde_lat, pde_longitude=pde_lon, mb=mb, ms=ms,
                   pde_depth_in_m=pde_depth_in_m, region_tag=region_tag,
                   eventname=eventname, cmt_time=cmt_time,
                   half_duration=half_duration, latitude=latitude,
                   longitude=longitude, depth_in_m=depth_in_m,
                   m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_rt=m_rt,
                   m_rp=m_rp, m_tp=m_tp)

    def to_row(self):
        """Returns a tuple of all parameters to append it to a 
        dataframe."""
        return (
            self.origin_time.datetime,
            self.pde_latitude,
            self.pde_longitude,
            self.pde_depth_in_m,
            self.mb,
            self.ms,
            self.region_tag,
            self.eventname,
            self.time_shift,
            self.half_duration,
            self.latitude,
            self.longitude,
            self.depth_in_m,
            self.m_rr,
            self.m_tt,
            self.m_pp,
            self.m_rt,
            self.m_rp,
            self.m_tp
        )

    def write_CMTSOLUTION_file(self, filename, mode="w"):
        """
        Initialize a source object from a CMTSOLUTION file.
        :param filename: path to the CMTSOLUTION file
        """
        with open(filename, mode) as f:
            # Reconstruct the first line as well as possible. All
            # hypocentral information is missing.
            f.write(
                " PDE %4i %2i %2i %2i %2i %5.2f %8.4f %9.4f %5.1f %.1f %.1f"
                " %s\n" % (
                    self.origin_time.year,
                    self.origin_time.month,
                    self.origin_time.day,
                    self.origin_time.hour,
                    self.origin_time.minute,
                    self.origin_time.second
                    + self.origin_time.microsecond / 1E6,
                    self.pde_latitude,
                    self.pde_longitude,
                    self.pde_depth_in_m / 1e3,
                    self.mb,
                    self.ms,
                    str(self.region_tag)))
            f.write('event name:     %s\n' % (str(self.eventname)))
            f.write('time shift:%12.4f\n' % (self.time_shift,))
            f.write('half duration:%9.4f\n' % (self.half_duration,))
            f.write('latitude:%14.4f\n' % (self.latitude,))
            f.write('longitude:%13.4f\n' % (self.longitude,))
            f.write('depth:%17.4f\n' % (self.depth_in_m / 1e3,))
            f.write('Mrr:%19.6e\n' % self.m_rr)  # * 1e7,))
            f.write('Mtt:%19.6e\n' % self.m_tt)  # * 1e7,))
            f.write('Mpp:%19.6e\n' % self.m_pp)  # * 1e7,))
            f.write('Mrt:%19.6e\n' % self.m_rt)  # * 1e7,))
            f.write('Mrp:%19.6e\n' % self.m_rp)  # * 1e7,))
            f.write('Mtp:%19.6e\n' % self.m_tp)  # * 1e7,))

    @property
    def M0(self):
        """
        Scalar Moment M0 in Nm
        """
        return (self.m_rr ** 2 + self.m_tt ** 2 + self.m_pp ** 2
                + 2 * self.m_rt ** 2 + 2 * self.m_rp ** 2
                + 2 * self.m_tp ** 2) ** 0.5 * 0.5 ** 0.5

    @M0.setter
    def M0(self, M0):
        """
        Scalar Moment M0 in Nm
        """
        iM0 = self.M0
        fM0 = M0
        factor = fM0/iM0
        self.m_rr *= factor
        self.m_tt *= factor
        self.m_pp *= factor
        self.m_rt *= factor
        self.m_rp *= factor
        self.m_tp *= factor

        self.update_hdur()

    @property
    def moment_magnitude(self):
        """
        Moment magnitude M_w
        """
        return 2/3 * np.log10(7 + self.M0) - 10.73  # =  (log Mo - 9.1) / 1.5 = (2/3) * (log Mo - 9.1)

    @property
    def time_shift(self):
        """
        Time shift between cmtsolution and pdesolution
        """
        return self.cmt_time - self.origin_time

    @time_shift.setter
    def time_shift(self, time_shift):
        self.cmt_time = self.origin_time + time_shift

    @property
    def tensor(self):
        """
        List of moment tensor components in r, theta, phi coordinates:
        [m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
        """
        return np.array([self.m_rr, self.m_tt, self.m_pp, self.m_rt, self.m_rp,
                         self.m_tp])

    @tensor.setter
    def tensor(self, tensor):
        """
        List of moment tensor components in r, theta, phi coordinates:
        [m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
        """
        self.m_rr = tensor[0]
        self.m_tt = tensor[1]
        self.m_pp = tensor[2]
        self.m_rt = tensor[3]
        self.m_rp = tensor[4]
        self.m_tp = tensor[5]

        # self.update_hdur()

    @property
    def fulltensor(self):
        """
        ndarray of full moment tensor components in r, theta, phi coordinates:
        """
        return np.array([[self.m_rr, self.m_rt, self.m_rp],
                         [self.m_rt, self.m_tt, self.m_tp],
                         [self.m_rp, self.m_tp, self.m_pp]])

    def update_hdur(self):
        # Updates the half duration
        Nm_conv = 1 / 1e7
        self.half_duration = np.round(
            2.26 * 10**(-6) * (self.M0 * Nm_conv)**(1/3), decimals=1)

    @property
    def pbt(self):
        """Returns tension (t), null (b), and pressure (p) axis and
        corresponding eigenvalues.

        Returns
        -------
        tuple
            matrix with corresponding eigenvalues, tbp column vectors
        """
        # Get eigenvalues and eigenvectors
        (lb, ev) = np.linalg.eigh(self.fulltensor)

        order = lb.argsort()  # in decreasing order -> pbt

        return lb[order], ev[order]

    @property
    def tnp_deg(self):
        """Returns tension (t), null (b), and pressure (p) axis in terms of 
        eignevalue, az, plunge

        Returns
        -------
        tuple
            matrix with corresponding eigenvalues, tbp column vectors
        """

        # Get eigenvalues and vectors
        (d, v) = self.pbt

        # compute plunges
        pl = np.arcsin(-v[0])

        # Compute azimuthes
        az = np.arctan2(v[2], -v[1])

        # Fix angles
        for i in range(0, 3):
            if pl[i] <= 0:
                pl[i] = -pl[i]
                az[i] += np.pi
            if az[i] < 0:
                az[i] += 2 * np.pi
            if az[i] > 2 * np.pi:
                az[i] -= 2 * np.pi
        
        # Radians to degress
        pl *= 180.0/np.pi
        az *= 180.0/np.pi

        # -> eignevalue, 
        t = np.array([d[2], az[2], pl[2]])
        n = np.array([d[1], az[1], pl[1]])
        p = np.array([d[0], az[0], pl[0]])
    
        return (t, n, p)

    @property
    def tbp_norm(self):
        """Returns the same as tpb, but eigenvalues are normalized
        by the scalar moment.
        """
        lb, ev = self.pbt
        return lb/self.M0, ev

    @property
    def fns(self):
        """
        Return fault normal and slip vectors
        """
        E, pbt = self.pbt

        # Get pressure and tension axes
        T, _, P = pbt[:, 2], pbt[:, 1], pbt[:, 0]

        # Get two directions
        TP1 = T+P
        TP2 = T-P

        # Get unit normals
        normal = (TP1)/np.sqrt(np.sum(TP1**2))
        slip = (TP2)/np.sqrt(np.sum(TP2**2))

        return normal, slip

    @property
    def eqpar(self):
        """For documentation see lwsspy.seismo.eqpar."""
        return list(eqpar(self.tensor))

    @property
    def sdr(self):
        """
        Returns
        -------
        (strikes, dips, rakes)
        """
        return self.eqpar[1:4]

    @ staticmethod
    def normal2sd(normal):
        """Compute strike and dip from normal

        Parameters
        ----------
        normal : [type]
            [description]
        """

        # strike
        strike = np.arctan2(normal[1], -normal[2])
        # strike = np.mod(strike, 2*np.pi)

        # dip
        dip = np.arctan2((normal[1]**2+normal[0]**2),
                         np.sqrt((normal[0]*normal[2])**2+(normal[1]*normal[2])**2))
        # dip = np.arccos(normal[2]/np.sqrt(np.sum(normal**2)))

        return strike, dip

    def beach(self):
        updaterc()
        plt.figure(figsize=(2, 2))
        ax = plt.axes()

        # Plot beach ball
        bb = beach(self.tensor,
                   linewidth=2,
                   facecolor='k',
                   bgcolor='w',
                   edgecolor='k',
                   alpha=1.0,
                   xy=(0.5, 0.5),
                   width=200,
                   size=100,
                   nofill=False,
                   zorder=100,
                   axes=ax)
        ax.add_collection(bb)
        
        # This fixes pdf output issue
        bb.set_transform(transforms.Affine2D(np.identity(3)))

        ax.axis('off')

    def axbeach(
        self, ax, x, y, width=50, facecolor='k', linewidth=2, alpha=1.0, 
        clip_on=False, **kwargs):
        """Plots beach ball into given axes.
        Note that width heavily depends on the given screen size/dpi. Therefore
        often does not work."""

        # Plot beach ball
        bb = beach(self.tensor,
                   linewidth=linewidth,
                   facecolor=facecolor,
                   bgcolor='w',
                   edgecolor='k',
                   alpha=alpha,
                   xy=(x, y),
                   width=width,
                   size=100, # Defines number of interpolation points 
                   axes=ax,
                   **kwargs)
        bb.set(clip_on=clip_on)

        # This fixes pdf output issue
        bb.set_transform(transforms.Affine2D(np.identity(3)))
        
        ax.add_collection(bb)

    def beachfig(self):
        """
        SDR
        M0
        location
        origin time
        centroid time shift
        half duration
        3x3 image black 1, white 0
        """
        updaterc()
        plt.figure(figsize=(5.25, 1.75))
        ax = plt.axes()
        ax.axis('off')

        # Plot beach ball
        bb = beach(self.tensor,
                   linewidth=1,
                   facecolor='k',
                   bgcolor='w',
                   edgecolor='k',
                   alpha=1.0,
                   xy=(0.625, 0.4),
                   width=145,
                   size=100,
                   nofill=False,
                   zorder=100,
                   axes=ax)
        ax.add_collection(bb)
        
        # This fixes pdf output issue
        bb.set_transform(transforms.Affine2D(np.identity(3)))

        # Base info string
        title_string = f'{self.eventname}'
        header_topleft = 'PDE:'

        topleft = '\n'
        topleft += f'Magnitudes: '
        topleft += f'Mw {self.moment_magnitude:4.2f}, '
        topleft += f'mb {self.mb:4.2f}, '
        topleft += f'ms {self.ms:4.2f}\n'
        topleft += f'Origin: {self.origin_time.strftime("%d-%m-%y %H:%M:%S"):>19}\n'
        topleft += f'Lat, Lon: {"":>1}{self.pde_latitude:>7.2f}, {self.pde_longitude:>7.2f}\n'
        topleft += f'Depth: {self.pde_depth_in_m/1000:>17.1f} km\n'

        bottomleft = ''
        bottomleft += f'Time Shift: {self.time_shift:>13} s\n'
        bottomleft += f'Lat, Lon: {"":>1}{self.latitude:>7.2f}, {self.longitude:>7.2f}\n'
        bottomleft += f'Depth: {self.depth_in_m/1000:>17.1f} km\n'
        bottomleft += f'hdur: {self.half_duration:>19} s'

        ss, ds, rs = self.sdr
        bottomright = ''
        bottomright += f'S/D/R:\n'
        bottomright += f'{ss[0]:3.0f}/{ds[0]:3.0f}/{rs[0]:4.0f}\n'
        bottomright += f'{ss[1]:3.0f}/{ds[1]:3.0f}/{rs[1]:4.0f}'

        # Topleft text
        plot_label(ax, header_topleft, location=1, dist=0.0, box=False,
                   fontdict=dict(family='monospace', size='x-small',
                                 fontweight='bold'))

        plot_label(ax, topleft, location=1, dist=0.0, box=False,
                   fontdict=dict(family='monospace', size='x-small'))

        # Bottom left text
        header_bottomleft = 'CMT:' + bottomleft.count('\n') * '\n' + '\n'

        plot_label(ax, header_bottomleft, location=3, dist=0.0, box=False,
                   fontdict=dict(family='monospace', size='x-small',
                                 fontweight='bold'))
        plot_label(ax, bottomleft, location=3, dist=0.0, box=False,
                   fontdict=dict(family='monospace', size='x-small'))

        # Bottom left text
        # header_bottomleft = 'CMT:' + bottomleft.count('\n') * '\n' + '\n'

        # plot_label(ax, header_bottomleft, location=3, dist=0.0, box=False,
        #            fontdict=dict(family='monospace', size='x-small',
        #                          fontweight='bold'))
        plot_label(ax, bottomright, location=4, dist=0.0, box=False,
                   fontdict=dict(family='monospace', size='x-small'))

        # Diverging Red to White to Black
        cmapname = 'RdGy'
        cmap = plt.get_cmap(cmapname)
        norm = MidpointNormalize(vmin=-1.0, midpoint=0.0, vmax=1.0)

        # Make the tensor scaled to between -1 to 1
        absmax = np.max(np.abs(self.tensor))
        mt = self.fulltensor/absmax

        # Get the aspect of the original and the new axes
        fraction = 0.3
        asp = get_aspect(ax)
        subax = axes_from_axes(
            ax, 123,
            extent=[1-fraction*asp, 1-fraction, fraction*asp, fraction])
        im = plt.imshow(mt, cmap=cmap, norm=norm)
        subax.axis('off')

        # Create axes for colorbar
        cfrac = 0.1
        cax = axes_from_axes(
            ax, 123, extent=[
                1 - fraction*asp*(1 + cfrac), 1-fraction,
                fraction*asp*cfrac, fraction])
        plt.colorbar(im, cax=cax)
        cax.axis('off')

    def decomp(self, dtype="eps_nu"):
        """Returns decomposition based on eignevalues of the moment tensor

        Parameters
        ----------
        dtype : str, optional
            type of decomposition. implement new decomposition by adding
            function to sourcedecomposition module, by default "eps_nu"

        Returns
        -------
        arraylike
            decomposed source, output depends on function output.

        Raises
        ------
        ValueError
            If dtype is not implemented an error will be raised.
        """

        # Get possible decompositions
        dtypes = [func for func, _ in getmembers(
            sourcedecomposition, isfunction)]

        if dtype not in dtypes:
            raise ValueError(
                f"{dtype} not implemented. Possible dtypes are: {dtypes}")

        # Get function from the module
        decompfunc = getattr(sourcedecomposition, dtype)
        (M3, M2, M1), _ = self.pbt

        return decompfunc(M1, M2, M3)

    def __str__(self):

        # Reconstruct the first line as well as possible. All
        # hypocentral information is missing.
        if isinstance(self.origin_time, UTCDateTime):
            return_str = \
                " PDE %4i %2i %2i %2i %2i %5.2f %8.4f %9.4f %5.1f %.1f %.1f" \
                " %s\n" % (
                    self.origin_time.year,
                    self.origin_time.month,
                    self.origin_time.day,
                    self.origin_time.hour,
                    self.origin_time.minute,
                    self.origin_time.second
                    + self.origin_time.microsecond / 1E6,
                    self.pde_latitude,
                    self.pde_longitude,
                    self.pde_depth_in_m / 1e3,
                    self.mb,
                    self.ms,
                    self.region_tag)
        else: 
            return_str = "----- CMT Delta: ------- \n"

        return_str += 'event name:  %10s\n' % (str(self.eventname),)
        return_str += 'time shift:%12.4f\n' % (self.time_shift,)
        return_str += 'half duration:%9.4f\n' % (self.half_duration,)
        return_str += 'latitude:%14.4f\n' % (self.latitude,)
        return_str += 'longitude:%13.4f\n' % (self.longitude,)
        return_str += 'depth:%17.4f\n' % (self.depth_in_m / 1e3,)
        return_str += 'Mrr:%19.6e\n' % self.m_rr  # * 1e7,))
        return_str += 'Mtt:%19.6e\n' % self.m_tt  # * 1e7,))
        return_str += 'Mpp:%19.6e\n' % self.m_pp  # * 1e7,))
        return_str += 'Mrt:%19.6e\n' % self.m_rt  # * 1e7,))
        return_str += 'Mrp:%19.6e\n' % self.m_rp  # * 1e7,))
        return_str += 'Mtp:%19.6e\n' % self.m_tp  # * 1e7,))
    
        return return_str

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        """ Making the class iterable through key,value pairs. """
        # first start by grabbing the Class items
        iters = self.__dict__.items()

        # now 'yield' through the items
        for x, y in iters:
            yield x

    def __getitem__(self, item):
        """ Making the CMT Source subscriptable with indeces."""
        return self.__dict__[item]

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ != other.__dict__

    @staticmethod
    def same_eventids(id1, id2):

        id1 = id1 if not id1[0].isalpha() else id1[1:]
        id2 = id2 if not id2[0].isalpha() else id2[1:]

        return id1 == id2

    def __sub__(self, other):
        """ USE WITH CAUTION!! 
        -> Origin time becomes float of delta t
        -> centroid time becomes float of delta t
        -> half duration is weird to compare like this as well.
        -> the other class will be subtracted from this one and the resulting 
           instance will keep the eventname and the region tag from this class
        """

        if not self.same_eventids(self.eventname, other.eventname):
            raise ValueError('CMTSource.eventname must be equal to compare the events')

        # The origin time is the most problematic part
        origin_time = self.origin_time - other.origin_time
        pde_latitude = self.pde_latitude - other.pde_latitude
        pde_longitude = self.pde_longitude - other.pde_longitude
        pde_depth_in_m = self.pde_depth_in_m - other.pde_depth_in_m
        region_tag = self.region_tag
        eventame = self.eventname
        mb = self.mb - other.mb
        ms = self.ms - other.ms
        cmt_time = self.cmt_time - other.cmt_time
        half_duration = self.half_duration - other.half_duration
        latitude = self.latitude - other.latitude
        longitude = self.longitude - other.longitude
        depth_in_m = self.depth_in_m - other.depth_in_m
        m_rr = self.m_rr - other.m_rr
        m_tt = self.m_tt - other.m_tt
        m_pp = self.m_pp - other.m_pp
        m_rt = self.m_rt - other.m_rt
        m_rp = self.m_rp - other.m_rp
        m_tp = self.m_tp - other.m_tp

        return CMTSource(
            origin_time=origin_time,
            pde_latitude=pde_latitude, pde_longitude=pde_longitude, mb=mb, ms=ms,
            pde_depth_in_m=pde_depth_in_m, region_tag=region_tag, 
            eventname=eventame, cmt_time=cmt_time, half_duration=half_duration, 
            latitude=latitude, longitude=longitude, depth_in_m=depth_in_m, 
            m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_rt=m_rt, m_rp=m_rp, m_tp=m_tp)


def plot_beach():
    cmtsource = CMTSource.from_CMTSOLUTION_file(sys.argv[1])
    cmtsource.beach()
    plt.show(block=True)


def plot_beachfig():
    cmtsource = CMTSource.from_CMTSOLUTION_file(sys.argv[1])
    cmtsource.beachfig()
    plt.show(block=True)


