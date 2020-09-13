'''
Basic hiders
'''
import numpy as np


class SingleBitHider:
    '''
    Hide a bit in one coefficient.

    Build an hider from a scanner and embedder:
    hider = SingleBitHider(scan, embeder)

    then you can insert a bit in a coefficient
    index = 0
    hider.insert(1, index)

    or extract a bit from a coefficient
    hider.extract(10)
    '''

    def __init__(self, scan, embedder):
        '''
        Initialize self. See help(type(self)) for accurate signature.
        '''
        self.embedder = embedder
        self.scan = scan

    def insert(self, cover, bit=1, index=0):
        '''
        Hide a bit

        Arguments:
        bit -- bit to hide
        index -- index of coefficient where bit will be hidden
        '''
        data = np.copy(cover)
        scanning = self.scan(data)
        amplitude = scanning[index]
        scanning[index] = self.embedder.embed(amplitude, bit)

        return data

    def extract(self, cover, index=0):
        '''
        Get bit hidden an return it

        Arguments:
        index -- index of coefficient where bit will be extracted
        '''
        scanning = self.scan(cover)
        amplitude = scanning[index]
        return self.embedder.extract(amplitude)


class TransformHider:
    '''
    Gives to an arbitrary hider the capacity
    to hide payload in transform domain using
    an arbitrary transform too.

    hider and transform dependencies are set in itialization:
    hider = TransformHider(base_hider, transform)

    This class implement hider interface
    hider.insert(cover_work, ...)
    hider.extract(ws_work, ....)

    Aditional arguments are pased to based hider.
    '''
    def __init__(self, hider, transform):
        '''
        Initialize self. See help(type(self)) for accurate signature.
        '''
        self.hider = hider
        self.transform = transform

    def insert(self, cover_work, **kwargs):
        '''
        Insert the payload in transform domain using
        base hider.
        '''
        transformed_cover_work = self.transform.direct(cover_work)
        transformed_ws_work = self.hider.insert(transformed_cover_work, **kwargs)
        ws_work = self.transform.inverse(transformed_ws_work)

        return ws_work

    def extract(self, ws_work, **kwargs):
        '''
        Extract payload from transform domain using
        base hider.
        '''
        transformed_ws_work = self.transform.direct(ws_work)
        msg = self.hider.extract(transformed_ws_work, **kwargs)

        return msg