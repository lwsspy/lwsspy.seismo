

from .split_stream_inv import split_stream_inv
from ...utils.timer import Timer
from .process import process_stream
from obspy import Stream
from mpi4py import MPI


class MPIProcessStream:
    st: Stream
    process_dict: dict

    def __init__(self):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def get_stream_and_processdict(self, stream: Stream, processdict: dict):
        """Loads stream and process dict. Has to be executed on rank 0

        Parameters
        ----------
        st : Stream
            [description]
        """
        if self.rank != 0:
            raise ValueError("Loading script executed not on rank 1")

        self.stream = stream
        self.processdict = processdict

    def process(self):

        if self.rank == 0:
            t = Timer()
            t.start()

            # Split the stream into different chunks
            streamlist, _ = split_stream_inv(
                self.stream, self.processdict['inventory'], nprocs=self.size)
            processdict = self.processdict

        else:
            streamlist = None
            processdict = None

        # Scatter stream chunks
        streamlist = self.comm.scatter(streamlist, root=0)

        # Broadcast process dictionary
        processdict = self.comm.bcast(processdict, root=0)

        print(
            f"Stream {len(streamlist)} -- "
            f"Inv: {len(processdict['inventory'].get_contents()['channels'])} -- "
            f"Rank: {self.rank}/{self.size}", flush=True)

        # Process
        results = []
        result = process_stream(streamlist, **processdict)

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

            self.processed_stream = resultst
