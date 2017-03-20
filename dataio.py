import pandas as pd
import numpy as np
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


class BaseSeriesIterator(object):
    '''

    :param df: data frame
    :param labels: list
    :param window: window length, int
    :param shuffle: Boolean
    :param limit: int

    '''
    def __init__(self, df, labels, window, shuffle=True, limit=None):
        self.df = df
        self.labels = list(labels)
        self.features = [col for col in df.columns if col not in self.labels]
        self.window = window
        self.shuffle = shuffle
        self.limit = limit
        self.len = len(df)
        self.indices = df.index

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


STICKER = "MAR_CODE"


def validate_index(df, index, window):
    start = index - window + 1
    if start < 0:
        return False
    if len(df[STICKER][start: index + 1].unique()) > 1:
        return False
    return True


class SeriesIterator(BaseSeriesIterator):

    def __init__(self, df, labels, window, shuffle=True, limit=None):
        super(SeriesIterator, self).__init__(df, labels, window, shuffle, limit)
        if self.shuffle:
            self.curr_id = self.rand_index()
        else:
            self.curr_id = window - 1
        assert self.curr_id >= 0

    def rand_index(self):
        while True:
            curr_id = np.random.randint(0, self.len)
            #if validate_index(self.df, curr_id, window=self.window):
            break
        return curr_id

    def reset(self):
        #self.curr_id = 0

        self.curr_id = self.window - 1
        assert self.curr_id >= 0

    def next(self):
        curr_id = self.curr_id
        #print("cur_id:", curr_id)
        if curr_id >= self.len:
            self.reset()
            raise StopIteration
        else:
            self.curr_id = self.rand_index() if self.shuffle else curr_id + 1
            df_win = self.df[curr_id - self.window + 1: curr_id + 1]
            labels = self.labels[0] if len(self.labels) == 1 else self.labels
            labels_val = self.df[labels].values[curr_id]
            features_values = df_win[self.features].values
            if features_values.shape[0] == 1:
                features_values = features_values.reshape(features_values.shape[1:])
            features_values = features_values.reshape(features_values.shape+(1,))
            #temp = np.expand_dims(features_values, axis=0), labels_val
            return features_values,labels_val
            #return np.expand_dims(features_values, axis=0), labels_val

    def __next__(self):
        return self.next()


#Not an efficient implementation of appending data into nparray, better off use list and convert it to ndarray later
@threadsafe_generator
def batch(iters, output_num=2, batch_size=2):
    iters.reset()
    while True:
        outs = [[] for _ in range(output_num)]
        for sample in iters:
            for i in range(output_num):
                outs[i].append(sample[i])
            if len(outs[0]) == batch_size:
                yield [np.array(out) for out in outs]
                outs = [[] for _ in range(output_num)]
        if len(outs[0]) > 0:
            yield [np.array(out) for out in outs]


def main():
    df = pd.DataFrame({"MAR_CODE": [1, 1, 1, 1, 2, 2, 2],
                       "HIGH": [11, 12, 13, 14, 15, 16, 17],
                       "OPEN": [17, 18, 19, 20, 21, 22, 23],
                       "CLOSE": [1, 2, 3, 4, 5, 6, 7]})
    print(df)
    data = SeriesIterator(df, ["CLOSE"], window=4, shuffle=False)
    # for _ in range(10):
    #     print(next(data))
    for x in data:
        print(x)
    pipe = batch(data, batch_size=2)
    for a,b in pipe:
        print(a, b)


if __name__ == '__main__':
    main()