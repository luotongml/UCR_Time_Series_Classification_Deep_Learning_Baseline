import threading


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def count():
    i = 0
    while True:
        i += 1
        yield i

def loop(func, n):
    """Runs the given function n times in a loop.
    """
    for i in range(n):
        func()

def run(f, repeats=1000, nthreads=10):
    """Starts multiple threads to execute the given function multiple
    times in each thread.
    """
    # create threads
    threads = [threading.Thread(target=loop, args=(f, repeats))
               for i in range(nthreads)]

    # start threads
    for t in threads:
        t.start()

    # wait for threads to finish
    for t in threads:
        t.join()

def main():
    c1 = count()
    c2 = count()

    # call c1.next 100K times in 2 different threads
    run(c1.next, repeats=100000, nthreads=2)
    print("c1", c1.next())

    # call c2.next 100K times in 2 different threads
    run(c2.next, repeats=100000, nthreads=2)
    print("c2", c2.next())

if __name__ == "__main__":
    main()