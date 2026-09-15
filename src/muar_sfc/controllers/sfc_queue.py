from queue import Queue
from time import monotonic as time


class Full(Exception):
    "Exception raised by Queue.put(block=0)/put_nowait()."

    pass


class SFCQueue(Queue):
    def __init__(self, max_size=0):
        Queue.__init__(self, max_size)
        # self.sfc_queue = []
        # self.number_of_sfcs = 0

    def put_sfc(self, sfc):
        self.put(sfc)

    def peek_sfc(self):
        return self.get()

    def _put_begin(self, item):
        self.queue.insert(0, item)

    def put_begin(self, item, block=True, timeout=None):
        """Put an item into the queue.

        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until a free slot is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Full exception if no free slot was available within that time.
        Otherwise ('block' is false), put an item on the queue if a free slot
        is immediately available, else raise the Full exception ('timeout'
        is ignored in that case).
        """
        with self.not_full:
            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        raise Full
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time() + timeout
                    while self._qsize() >= self.maxsize:
                        remaining = endtime - time()
                        if remaining <= 0.0:
                            raise Full
                        self.not_full.wait(remaining)
            self._put_begin(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


if __name__ == "__main__":
    q = SFCQueue()
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    q.put_sfc(a)
    q.put_sfc(b)
    q.put_begin(d)
    q.put_begin(e)
    q.put_sfc(c)

    print(q.peek_sfc())
    print(q.peek_sfc())
    print(q.peek_sfc())

    print(q.empty())
