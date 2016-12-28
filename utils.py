import numpy as np
import timeit

def softmax(x):
    '''
    Numerical stable version of softmax
    '''
    e_x = np.exp(x - np.max(x))
    #return e_x / e_x.sum(axis=1)
    return e_x / e_x.sum(axis=0)

def one_hot(x,M):
    x_onehot = np.zeros((len(x), M))
    x_onehot[np.arange(len(x)),x] = 1
    return x_onehot.astype(np.int64)

class time_it:
    def __enter__(self):
        self.start_time = timeit.default_timer()
        return None

    def __exit__(self, type, value, traceback):
        print 'Time Cost = %f seconds'%(timeit.default_timer() - self.start_time)


