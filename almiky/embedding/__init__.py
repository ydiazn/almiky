'''Embedders'''

from abc import ABC, abstractmethod


class Embedder(ABC):

    @abstractmethod
    def embed(amplitude, bit):
        '''Embed a bit in signal'''

    @abstractmethod
    def extract(amplitude):
        '''Extract a bit in signal'''
