# %%
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.widgets
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Slider, Button
from cartopy.crs import PlateCarree, Mollweide
from pyproj import exceptions
from lwsspy.seismo.compare_catalogs import CompareCatalogs
from obspy.imaging.beachball import beach

from lwsspy.maps import plot_map
from lwsspy.seismo import plot_quakes
from lwsspy.seismo.cmt_catalog import CMTCatalog
from lwsspy.plot.markerupdater import MarkerUpdater
import lwsspy.maps as lmap
import lwsspy.plot as lplt
from matplotlib.colors import Normalize

import logging
# %%


# %%


class CatalogExplore:

    def __init__(
        self, cc: CompareCatalogs,
        olddatabase: str = None,
        newdatabase: str = None
    ):
        self.loglevel = logging.DEBUG
        self.__setup_logger__()
        self.color = 'k'
        self.marker = 'v'

        self.cc = cc
        self.pc = PlateCarree()
        self.central_longitude = 0.0

        self.depth = self.cc.new.getvals(vtype="depth_in_m")/1000.0
        self.isort = np.argsort(self.depth)[::-1]
        self.depth = self.depth[self.isort]

        self.ids = self.cc.new.getvals(vtype="eventname")[self.isort]
        self.mw = self.cc.new.getvals(vtype="moment_magnitude")[self.isort]
        self.m0 = self.cc.new.getvals(vtype="M0")[self.isort]
        self.latitude = self.cc.new.getvals(vtype="latitude")[self.isort]
        self.longitude = self.cc.new.getvals(vtype="longitude")[self.isort]

        # old
        self.odepth = self.cc.old.getvals(vtype="depth_in_m")[self.isort]/1000
        self.omw = self.cc.old.getvals(vtype="moment_magnitude")[self.isort]
        self.om0 = self.cc.old.getvals(vtype="M0")[self.isort]
        self.olatitude = self.cc.old.getvals(vtype="latitude")[self.isort]
        self.olongitude = self.cc.old.getvals(vtype="longitude")[self.isort]

        self.origin_time = self.cc.new.getvals(vtype="origin_time")

        # Location Range
        self.xmin0, self.xmax0 = -180.0, 180.0
        self.ymin0, self.ymax0 = -90.0, 90.0
        self.xmin, self.xmax = self.xmin0, self.xmax0
        self.ymin, self.ymax = self.ymin0, self.ymax0
        self.xmin_map, self.xmax_map = self.xmin0, self.xmax0
        self.ymin_map, self.ymax_map = self.ymin0, self.ymax0
        self.extent0 = [self.xmin0, self.xmax0, self.ymin0, self.ymax0]
        self.extent = [self.xmin_map, self.xmax_map,
                       self.ymin_map, self.ymax_map]

        # Moment magnitude range
        self.mwmin0, self.mwmax0 = np.min(self.mw), np.max(self.mw)
        self.mwmin, self.mwmax = self.mwmin0, self.mwmax0

        # Depth Range
        self.zmin0, self.zmax0 = np.min(self.depth), np.max(self.depth)
        self.zmin, self.zmax = self.zmin0, self.zmax0

        # origin time range
        self.tmin0, self.tmax0 = np.min(
            self.origin_time), np.max(self.origin_time)
        self.tmin, self.tmax = self.tmin0, self.tmax0

        # For scatter size
        self.factor = 20

        self.nbins = 100

        self.extentchange_ax = None

        # Database locations
        self.old_database = olddatabase
        self.new_database = newdatabase

        # Do the main things
        self._setup_figure()
        self._create_sliders()
        self._create_buttons()
        self._plot_maps()
        self._update_plots()
        self._mpl_connect()

        plt.show(block=True)

    def _setup_figure(self):

        # Overall Gridspec
        gs = GridSpec(
            ncols=3, nrows=2,
            width_ratios=(1, 1, 0.6), wspace=0.05,
            height_ratios=(1, 1), hspace=0.1)

        # Right side outline with buffer
        subgs1 = GridSpecFromSubplotSpec(
            3, 3, gs[:, 2], height_ratios=[1, 20000, 1], width_ratios=[2, 20, 5],
        )

        # Right side outline without buffer
        subgs2 = GridSpecFromSubplotSpec(
            3, 3, gs[:, 2], height_ratios=[1, 20000, 1], width_ratios=[1, 200, 1],
        )

        # Space for sliders with buffer
        slidergs = GridSpecFromSubplotSpec(
            6, 2, subgs1[1, 1], height_ratios=[1, 1, 1, 1, 10, 40], hspace=0.25)

        # Space for buttons
        buttongs = GridSpecFromSubplotSpec(3, 5, slidergs[4, :], hspace=0.75)

        # Space for event without buffer
        eventgs = GridSpecFromSubplotSpec(
            6, 2, subgs2[1, 1], height_ratios=[1, 1, 1, 1, 10, 40], hspace=0.25)

        # This enables resizing of the scatter points when zooming
        self.mupdater = MarkerUpdater()

        # Figure boundaries
        self.fig = plt.figure(figsize=(18, 9))
        plt.subplots_adjust(left=0.025, right=0.99, bottom=0.025, top=0.975)

        # Axes for the events
        self.ax_quak = self.fig.add_subplot(
            gs[0, 0], zorder=100, projection=PlateCarree(
                central_longitude=self.central_longitude))

        # Axes for the depth change
        self.ax_ddep = self.fig.add_subplot(
            gs[0, 1], sharex=self.ax_quak, sharey=self.ax_quak,
            projection=PlateCarree(central_longitude=self.central_longitude))

        # Axes for the moment change
        self.ax_dmom = self.fig.add_subplot(
            gs[1, 0], sharex=self.ax_quak, sharey=self.ax_quak,
            projection=PlateCarree(central_longitude=self.central_longitude))

        # Axes for the horizontal location change
        self.ax_dloc = self.fig.add_subplot(
            gs[1, 1], sharex=self.ax_quak, sharey=self.ax_quak,
            projection=PlateCarree(central_longitude=self.central_longitude))

        # List of the axes
        self.maps = [self.ax_quak, self.ax_ddep, self.ax_dmom, self.ax_dloc]

        # Axes for the horizontal location change
        self.ax_gui = self.fig.add_subplot(
            gs[:, 2], facecolor=(0.9, 0.9, 0.9))
        self.ax_gui.tick_params(
            axis='both', which='both', bottom=0, left=0, top=0, right=0,
            labelbottom=0, labelleft=0)

        # Define slider Axes
        self.ax_slider_mom = self.fig.add_subplot(slidergs[0, :])
        self.ax_slider_dep = self.fig.add_subplot(slidergs[1, :])
        self.ax_slider_lat = self.fig.add_subplot(slidergs[2, :])
        self.ax_slider_lon = self.fig.add_subplot(slidergs[3, :])
        # self.ax_slider_tim = self.fig.add_subplot(guigs[4, :])

        # Define Button axes
        if self.old_database and self.new_database:
            self.ax_dist_button = self.fig.add_subplot(buttongs[0, :])
            self.ax_meas_button = self.fig.add_subplot(buttongs[1, :])

        self.ax_event = self.fig.add_subplot(
            eventgs[5, :], facecolor=(0.9, 0.9, 0.9))
        # self.ax_event.axis('off')
        self.ax_event.tick_params(
            axis='both', which='both', bottom=0, left=0, top=0, right=0,
            labelbottom=0, labelleft=0)

    def _mpl_connect(self):
        self.xrange.on_changed(self.updatexrange)
        self.yrange.on_changed(self.updateyrange)
        self.zrange.on_changed(self.updatezrange)
        self.mwrange.on_changed(self.updatemwrange)

        if self.old_database and self.new_database:
            self.bdist.on_clicked(self.onbuttonclick_station_dist)
            self.bdist.on_clicked(self.onbuttonclick_measurements)

        self.fig.canvas.mpl_connect('pick_event', self.onpick)

        # Update size of the markers on zoom
        self.mupdater.add_ax(self.ax_quak, ['size'])
        self.mupdater.add_ax(self.ax_ddep, ['size'])
        self.mupdater.add_ax(self.ax_dmom, ['size'])
        self.mupdater.add_ax(self.ax_dloc, ['size'])

        for _ax in self.maps:
            _ax.callbacks.connect('xlim_changed', self.on_xlims_change)
            _ax.callbacks.connect('ylim_changed', self.on_ylims_change)

    def _create_sliders(self):

        # Longitude Slider
        self.xrange = RangeSlider(
            self.ax_slider_lon, 'Lon',
            valmin=-180.0, valmax=180.0, valinit=[self.xmin, self.xmax], valfmt=None,
            closedmin=True, closedmax=True, dragging=True,
            valstep=None, orientation='horizontal')
        self.xrange.valtext.set_size('x-small')

        # Latitude Slider
        self.yrange = RangeSlider(
            self.ax_slider_lat, 'Lat',
            valmin=-90.0, valmax=90.0, valinit=[self.ymin, self.ymax],
            valfmt=None, closedmin=True, closedmax=True, dragging=True,
            valstep=None, orientation='horizontal')
        self.yrange.valtext.set_size('x-small')

        # Depth Slider
        self.zrange = RangeSlider(
            self.ax_slider_dep, 'Dep',
            valmin=0.0, valmax=800.0, valinit=[self.zmin, self.zmax],
            valfmt=None, closedmin=True, closedmax=True, dragging=True,
            valstep=None, orientation='horizontal')
        self.zrange.valtext.set_size('x-small')

        # Moment slider
        self.mwrange = RangeSlider(
            self.ax_slider_mom, 'Mw',
            valmin=0.0, valmax=10.0, valinit=[self.mwmin, self.mwmax],
            valfmt=None, closedmin=True, closedmax=True, dragging=True,
            valstep=None, orientation='horizontal')
        self.mwrange.valtext.set_size('x-small')

    def _create_buttons(self):

        if (self.old_database is None) or (self.new_database is None):
            return

        # lplt.plot_label(
        #     self.ax_dist_button, 'Station Distribution', box=False,
        #     location=15)
        self.bdist = Button(
            self.ax_dist_button, 'Station Distribution',
            color='0.95', hovercolor='0.7')
        self.bmeas = Button(
            self.ax_meas_button, 'Measurements',
            color='0.95', hovercolor='0.7')

    def _plot_maps(self):

        # Plot maps
        self._plot_quakes()
        self._plot_ddep()
        self._plot_dmom()
        self._plot_dloc()

        # Set range
        self.ax_quak.set_extent(self.extent0, crs=PlateCarree())
        self.ax_ddep.set_extent(self.extent0, crs=PlateCarree())
        self.ax_dmom.set_extent(self.extent0, crs=PlateCarree())
        self.ax_dloc.set_extent(self.extent0, crs=PlateCarree())

    def _plot_quakes(self):

        plt.sca(self.ax_quak)
        plot_map()
        self.cmt_scatter, _, _, _ = plot_quakes(
            self.latitude, self.longitude, self.depth,
            self.mw, ax=self.ax_quak, cmap='rainbow_r', legend=False,
            yoffsetlegend2=0.09, sizefunc=self.sizefunc)

    def _plot_ddep(self):

        ddep = (self.depth - self.odepth)/self.om0 * 100.0

        # Setup color range
        vmin, vcenter, vmax = np.quantile(
            ddep, 0.01), 0, np.quantile(ddep, 0.99)
        norm = lplt.MidpointNormalize(
            vmin=vmin, midpoint=vcenter, vmax=vmax)
        cmap = plt.get_cmap('seismic')

        # Plot
        plt.sca(self.ax_ddep)
        plot_map()
        self.ddep_scatter = self.ax_ddep.scatter(
            self.longitude, self.latitude,
            s=30*self.factor*np.abs(ddep)/np.max(np.abs(ddep)),
            c=ddep, transform=self.pc,
            cmap=cmap, alpha=1.0, norm=norm, edgecolor='k',
            linewidth=0.1, picker=True)

        # Set title ... gotta change that
        self.ax_ddep.set_title('dz')

    def _plot_dloc(self):

        # Compute the necessary thingies
        dlat = self.latitude - self.olatitude
        dlon = self.longitude - self.olongitude
        ddeg = lmap.haversine(
            self.olongitude, self.olatitude,
            self.longitude, self.latitude)
        b = 90 - lmap.bearing(
            self.olongitude, self.olatitude,
            self.longitude, self.latitude
        )

        # Setup color map
        vmin, vmax = 0, np.max(ddeg)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('inferno')

        # Plot
        plt.sca(self.ax_dloc)
        plot_map()
        self.loc_quiver = self.ax_dloc.quiver(
            self.olongitude, self.olatitude,
            dlon, dlat, ddeg, angles=b,
            pivot='tail', cmap=cmap, norm=norm, scale=0.025,
            transform=PlateCarree(), units='xy',  # width=100,width=0.005,
            linewidth=0.1, edgecolor='k', picker=True
        )
        km = 20
        deg = km/111.11
        qk = self.ax_dloc.quiverkey(self.loc_quiver, 0.9, 1.025, deg, rf'${20}\, km$', labelpos='E',
                                    coordinates='axes')
        # Setup title...
        self.ax_dloc.set_title('dloc')

    def _plot_dmom(self):

        # Compute change
        dm0 = (self.m0 - self.om0)/self.om0 * 100.0

        # Create colormap
        vmin, vcenter, vmax = np.min(dm0), 0, np.max(dm0)
        norm = lplt.MidpointNormalize(
            vmin=vmin, midpoint=vcenter, vmax=vmax)
        cmap = plt.get_cmap('seismic')

        # Plot
        plt.sca(self.ax_dmom)
        plot_map()
        self.dmom_scatter = self.ax_dmom.scatter(
            self.longitude, self.latitude,
            s=self.factor*np.abs(dm0)/np.max(np.abs(dm0)),
            c=dm0, transform=self.pc,
            cmap=cmap, alpha=1.0, norm=norm, edgecolor='k',
            linewidth=0.1, picker=True)

        # Set title
        self.ax_dmom.set_title('dlnM0')

    def __setup_logger__(self):

        # create logger
        self.logger = logging.getLogger('CatalogExplore')
        self.logger.setLevel(self.loglevel)

        # create console handler and set level to debug
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s | %(levelname)8s: '
            '%(message)s (%(filename)s:%(lineno)d)',
            datefmt='%m/%d/%Y %I:%M:%S %p')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        if len(self.logger.handlers) > 0:
            self.logger.handlers = []

        self.logger.addHandler(ch)

    def sizefunc(self, x):
        return np.pi*(0.25*(x-self.mwmin0)/(self.mwmax0-self.mwmin0) + 1)**8

    def filter_events(self):

        # Check which events are in the current range
        self.idx_bool = (self.xmin <= self.longitude) & (self.longitude <= self.xmax) & \
            (self.ymin <= self.latitude) & (self.latitude <= self.ymax) & \
            (self.zmin <= self.depth) & (self.depth <= self.zmax) & \
            (self.mwmin <= self.mw) & (self.mw <= self.mwmax)
        # & \
        # (self.tmin <= self.origin_time) & (self.origin_time <= self.tmax)

        self.idx = np.where(
            self.idx_bool
        )[0]

        self.logger.debug(
            f"Filter Events: {np.sum(self.idx_bool)}/{len(self.idx_bool)}")

        self.idx_float = self.idx_bool.astype(float)

    def on_xlims_change(self, event_ax):
        self.on_extent_change(event_ax)

    def on_ylims_change(self, event_ax):
        self.on_extent_change(event_ax)

    def on_extent_change(self, event_ax):

        if self.extentchange_ax:
            return

        self.extentchange_ax = event_ax

        xlim = event_ax.get_xlim()
        ylim = event_ax.get_ylim()
        self.logger.debug(f"updated xlims: {xlim}")
        self.logger.debug(f"updated ylims: {ylim}")

        self.updatexrange_map(xlim)
        self.updateyrange_map(ylim)
        self.update_extents()

        self.extentchange_ax = None

    def updatexrange(self, val):

        self.xmin, self.xmax = val[0], val[1]
        self.logger.debug(f"X range: [{self.xmin}, {self.xmax}]")
        self._update_plots()

    def updateyrange(self, val):

        self.ymin, self.ymax = val[0], val[1]
        self.logger.debug(f"Y range: [{self.ymin}, {self.ymax}]")
        self._update_plots()

    def updatexrange_map(self, val):
        self.xmin_map, self.xmax_map = val[0], val[1]
        self.extent[:2] = val[0], val[1]
        self.logger.debug(f"X range Map: [{self.xmin_map}, {self.xmax_map}]")

    def updateyrange_map(self, val):
        self.ymin_map, self.ymax_map = val[0], val[1]
        self.extent[2:] = val[0], val[1]
        self.logger.debug(f"Y range Map: [{self.ymin_map}, {self.ymax_map}]")

    def updatezrange(self, val):

        self.zmin, self.zmax = val[0], val[1]
        self.logger.debug(f"Z range: [{self.zmin}, {self.zmax}]")
        self._update_plots()

    def updatemwrange(self, val):

        self.mwmin, self.mwmax = val[0], val[1]
        self.logger.debug(f"Mw range: [{self.mwmin}, {self.mwmax}]")
        self._update_plots()

    def update_extents(self):

        self.logger.debug(f"Extent: {self.extent}")

        # Update axes
        for ax in self.maps:
            ax.set_extent(self.extent, self.pc)

        self.extentchanged = False

    def _update_plots(self):

        self.filter_events()

        self.cmt_scatter.set_alpha(self.idx_float)
        self.dmom_scatter.set_alpha(self.idx_float)
        self.ddep_scatter.set_alpha(self.idx_float)
        self.loc_quiver.set_alpha(self.idx_float)

        self.fig.canvas.draw()

    def onbuttonclick_station_dist(self, event):
        self.logger.debug('Button station dist')

        if not self.selected_event:
            return

        ocmtdir = os.path.join(
            self.old_database, self.ids[self.selected_event])
        ncmtdir = os.path.join(
            self.new_database, self.ids[self.selected_event])

        # Read measurement file
        # Plot things
        plt.figure(figsize=(8, 4))
        ax = plt.axes(projection=Mollweide(
            central_longitude=self.longitude[self.selected_event]))
        ax.gridlines()
        lmap.plot_map()
        # ax.plot(lon, lat, 'v', label="Stations", markeredgecolor='k',
        #         markerfacecolor=(0.8, 0.3, 0.3), transform=PlateCarree())
        # ax.set_extent(extent)

    def onbuttonclick_measurements(self, event):
        self.logger.debug('Button station dist')

    def onpick(self, event):
        ind = event.ind

        visind = [
            _ind for _ind in ind if event.artist.get_alpha()[_ind] == 1.0
        ]

        self.logger.debug(
            f'onpick3 scatter ind:\n'
            f'{ind}\n'
            f'{self.longitude[ind]}\n'
            f'{self.latitude[ind]}')
        self.logger.debug(
            f'onpick3 scatter visind:\n'
            f'{visind}\n'
            f'{self.longitude[visind]}\n'
            f'{self.latitude[visind]}')

        if len(visind) > 1:
            self.ax_event.set_title(
                f'{len(ind)} points overlapping.\nPlease Select single point.',
                size='x-small')

            self.selected_event = None

        else:
            self.ax_event.clear()
            self.ax_event.set_title(
                f"{self.ids[visind[0]]}")

            self.selected_event = visind[0]

            self.ax_event.set_xlim(0, 1)
            self.ax_event.set_ylim(0, 1)
            self.plot_beaches()
            self.plot_data()

        self.fig.canvas.draw()

    def plot_data(self):
        i = self.selected_event
        otshift = self.cc.old.getvals(vtype="time_shift")[i]
        ntshift = self.cc.old.getvals(vtype="time_shift")[i]

        stats = '\n'.join([
            'ID: {:s}'.format(self.ids[i]),
            '---------------------------------------',
            'Val:     GCMT ->  GCMT3D     *Delta    ',
            '---------------------------------------',
            'Mw:   {:7.2f} -> {:7.2f}    {:7.2f} %'.format(self.omw[i],
                                                           self.mw[i],
                                                           (self.m0[i] -
                                                            self.om0[i])
                                                           / self.om0[i]),
            '$\\theta$:    {:7.2f} -> {:7.2f}    {:7.2f} deg'.format(self.olatitude[i],
                                                                     self.latitude[i],
                                                                     self.latitude[i]
                                                                     - self.olatitude[i]),
            '$\\phi$:    {:7.2f} -> {:7.2f}    {:7.2f} deg'.format(self.olongitude[i],
                                                                   self.longitude[i],
                                                                   self.longitude[i]
                                                                   - self.olongitude[i]),
            'z:    {:7.2f} -> {:7.2f}    {:7.2f} km'.format(self.odepth[i],
                                                            self.depth[i],
                                                            self.depth[i]
                                                            - self.odepth[i]),
            'Ts:   {:7.2f} -> {:7.2f}    {:7.2f} s'.format(otshift,
                                                           ntshift,
                                                           ntshift
                                                           - otshift)
        ])
        lplt.plot_label(
            self.ax_event, stats, location=3, box=False,
            fontdict=dict(family="monospace", fontsize='small'),
            zorder=100000)

    def plot_beaches(self):

        self.logger.debug("Plotting Beachballs")

        otensor = self.cc.old.getvals()[self.selected_event]
        ntensor = self.cc.new.getvals()[self.selected_event]

        ybeach = 0.85
        xbeach = 0.2
        width = 200
        self.beach(
            otensor, xy=(xbeach, ybeach), width=width, color='k')
        self.beach(
            ntensor, xy=(1-xbeach, ybeach), width=width, color=(0.2, 0.2, 0.8))

        self.ax_event.arrow(0.475, ybeach, 0.05, 0, color='k', width=0.005)
        self.ax_event.figure.canvas.draw()

    def beach(self, tensor, xy=(0.5, 0.5), width=400, color='k'):

        # Plot beach ball
        bb = beach(tensor,
                   linewidth=2,
                   facecolor=color,
                   bgcolor='w',
                   edgecolor='k',
                   alpha=1.0,
                   xy=xy,
                   width=width,
                   size=100,
                   nofill=False,
                   zorder=100,
                   axes=self.ax_event)
        self.ax_event.add_collection(bb)


def bin():

    oldfile = "/Users/lucassawade/stats/final/GCMT.pkl"
    newfile = "/Users/lucassawade/stats/final/GCMT3D+X.pkl"

    old = CMTCatalog.load(oldfile)
    new = CMTCatalog.load(newfile)
    oldlabel = 'GCMT'
    newlabel = 'GCMT3D+X'

    # Compare Catalog
    CC = CompareCatalogs(old=old, new=new,
                         oldlabel=oldlabel, newlabel=newlabel,
                         nbins=25)

    CC, CC_pop = CC.filter(maxdict={"M0": 1.5, "latitude": 0.4,
                                    "longitude": 0.4, "depth_in_m": 30000.0})

    ce = CatalogExplore(CC)
