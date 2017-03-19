import pandas as pd
import numpy as np


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

    def __init__(self, df, labels, window, batch_size=32, output_num=2, shuffle=True, limit=None):
        super(SeriesIterator, self).__init__(df, labels, window, shuffle, limit)
        self.batch_size = batch_size
        self.output_num = 2
        if self.shuffle:
            self.curr_id = self.rand_index()
        else:
            self.curr_id = window - 1

    def rand_index(self):
        while True:
            curr_id = np.random.randint(0, self.len)
            if validate_index(self.df, curr_id, window=self.window):
                break
        return curr_id

    def reset(self):
        self.curr_id = 0

    def next(self):
        outs = [[] for _ in range(self.output_num)]
        while True:
            sample=self.next_one()
            if sample == None:
                if len(outs[0]) >0:
                    return [np.array(out) for out in outs]
                else:
                    self.reset()
                    raise StopIteration
            for i in range(self.output_num):
                outs[i].append(sample[i])
            if len(outs[0]) == self.batch_size:
                break
        #if len(outs[0]) > 0:
        return [np.array(out) for out in outs]

    def next_one(self):
        curr_id = self.curr_id
        if curr_id >= self.len:
            return None
        else:
            self.curr_id = self.rand_index() if self.shuffle else curr_id + 1
            df_win = self.df[curr_id - self.window + 1: curr_id + 1]
            labels = self.labels[0] if len(self.labels) == 1 else self.labels
            labels_val = self.df[labels].values[curr_id]
            features_values = df_win[self.features].values
            return np.expand_dims(features_values, axis=0), labels_val

    def __next__(self):
        return self.next()



def main():
    df = pd.DataFrame({"MAR_CODE": [1, 1, 1, 1, 2, 2, 2],
                       "HIGH": [11, 12, 13, 14, 15, 16, 17],
                       "OPEN": [17, 18, 19, 20, 21, 22, 23],
                       "CLOSE": [1, 2, 3, 4, 5, 6, 7]})
    print(df)
    data = SeriesIterator(df, ["CLOSE"], window=4, shuffle=False, batch_size=3)
    # for _ in range(10):
    #     print(next(data))
    for x in data:
        print(x)
    #pipe = batch(data, batch_size=3)
    #for x,y in pipe:
     #   print(x, y)


if __name__ == '__main__':
    main()