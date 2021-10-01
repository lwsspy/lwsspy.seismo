from obspy import Stream
from .window import window_on_stream
from ...utils.timer import Timer
from ..process.split_stream_inv import split_stream_inv
from mpi4py import MPI


class MPIWindowStream:
    obsd: Stream
    synt: Stream
    process_dict: dict

    def __init__(self):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def get_streams_and_windowdict(self, obsd: Stream, synt: Stream, windowdict: dict):
        """Loads stream and process dict. Has to be executed on rank 0

        Parameters
        ----------
        obsd : Stream
            observed data
        synt : Stream
            synthetic data
        """

        if self.rank != 0:
            raise ValueError("Loading script executed not on rank 1")

        self.obsd = obsd
        self.synt = synt
        self.windowdict = windowdict

    def window(self):

        if self.rank == 0:
            t = Timer()
            t.start()

            # Split the stream into different chunks
            obsdlist, syntlist, _ = split_stream_inv(
                self.obsd, self.windowdict['station'], synt=self.synt, nprocs=self.size)
            windowdict = self.windowdict

        else:
            obsdlist = None
            syntlist = None
            windowdict = None

        # Scatter stream chunks
        obsdlist = self.comm.scatter(obsdlist, root=0)
        syntlist = self.comm.scatter(syntlist, root=0)

        # Broadcast process dictionary
        windowdict = self.comm.bcast(windowdict, root=0)

        # Get stream and inventory chunks
        print(
            f"Observed {len(obsdlist)} -- Synthetic {len(syntlist)} --"
            f"Inv: {len(windowdict['station'].get_contents()['channels'])} -- "
            f"Rank: {self.rank}/{self.size}", flush=True)

        # Process
        results = []
        result = window_on_stream(obsdlist, syntlist, **windowdict)
        # for _tr in result:
        #     try:
        #         print(
        #             f"Rank: {self.rank} - {_tr.id} - {len(_tr.stats.windows):3d}")
        #     except Exception:
        #         print(f"----- {_tr.id}")
        results.append(result)
        print(f"Rank: {self.rank}/{self.size} -- Done.", flush=True)

        results = self.comm.gather(results, root=0)

        # Sort
        if self.rank == 0:
            # Flatten list of lists.
            resultst = Stream()
            for _result in results:
                resultst += _result[0]

            t.stop()
            self.windowed_stream = resultst
