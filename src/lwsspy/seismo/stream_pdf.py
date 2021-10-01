import datetime
from typing import Optional
from .plot_seismogram import plot_seismogram
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from obspy import Stream
from .source import CMTSource


def stream_pdf(
        obsd: Stream,
        synt: Optional[Stream] = None,
        cmtsource: Optional[CMTSource] = None,
        tag: Optional[CMTSource] = None,
        outfile: str = "stream.pdf"):

    backend = plt.get_backend()
    plt.switch_backend("pdf")

    if 'distance' in obsd[0].stats:
        obsd.sort('distance')

    with PdfPages(outfile) as pdf:
        for _obsd_tr in obsd:

            # Get sunthetic trace if synthetics are there
            if synt is not None:
                try:
                    synt_tr = synt.select(
                        station=_obsd_tr.stats.station,
                        network=_obsd_tr.stats.network,
                        component=_obsd_tr.stats.channel[-1])[0]
                except Exception as err:
                    print("Couldn't find corresponding synt for obsd trace(%s):"
                          "%s" % (_obsd_tr.id, err))
                    continue
            else:
                synt_tr = None

            # plot seismograms and windows
            fig = plot_seismogram(
                _obsd_tr, synt=synt_tr, cmtsource=cmtsource, tag=tag)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close(fig)

        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = f"Seismic-Wave-Data-PDF"
        d['Author'] = 'Lucas Sawade'
        d['Subject'] = 'Trace comparison in one pdf'
        d['Keywords'] = 'seismology, moment tensor inversion'
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()

    plt.switch_backend(backend)
