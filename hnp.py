from Crypto.Util.number import *
import random
import os
import re
from collections import namedtuple
from sympy import prevprime


Eq = namedtuple('Eq', ['a', 'b', 'k'])


def getMask(mask):
    """
    # is known   ? is unknown
    for a 32 bit number
    '31 # 20 10 # 5' means:
    ########## ?????????? ##### ?????
    """
    res = []
    for _ in re.findall(r'(\d+) # (\d+)', mask):
        _ = [int(_[1]), int(_[0]) - int(_[1])]
        assert _[1] >= 0
        res.append((_[0], _[1], (1 << _[1]) - 1 << _[0]))
    return res


class HNP:
    """
    Hidden Number Problem (HNP) generator
    a * x = b mod p
    let # be known part, ? unknown
    you can generate by mask:
        ######????????????? for range(n)
        ????#####?????????? for range(n)
        ???####?????####??? for range(n)
    and you can use different known part for each equation:
        ###???####???????## for 0
        ######????##??###?? for 1
            ...
        ###???##?###??###?? for n - 1
    """
    def __init__(self, m: int, n: int, known=None, mask=None, prime=True, seed=None, prime_full=False):
        """
        each equation contains (a, b, k)
        b = a * x % p
        k is known part
        :param m: the bit length of the module
        :param n: the number of equations
        :param known: Optional, if not None: will generate the k-MSB of a * x
        :param mask: Optional, if not None: generate known part(s) by user's input
        :param prime: Optional, if False: the module will be set by 2^m
        :param seed: Optional, if is not None: use random set by seed
        """
        self.m = m
        self.n = n

        if seed is None:
            self.randfunc = os.urandom
        else:
            random.seed(seed)
            self.randfunc = random.randbytes

        if prime:
            # this will leak more information in MSB
            if prime_full:
                self.q = prevprime(1 << m)
            else:
                self.q = getPrime(m, self.randfunc)
        else:
            self.q = 1 << m

        # you find the Hidden Number :)
        self.x = getRandomRange(0, self.q, self.randfunc)

        self.eqs = []
        for _ in range(n):
            a = getRandomRange(0, self.q, self.randfunc)
            b = a * self.x % self.q
            self.eqs.append(Eq(a, b, None))

        if known is not None:
            assert not mask, 'should not give the mask'
            self.known = known
            self.eqs = [eq._replace(k=eq.b >> (self.m - self.known) << (self.m - self.known)) for eq in self.eqs]
        else:
            assert mask, 'must give known bits or mask'
            if type(mask) is str:
                self.mask = getMask(mask)
                self.eqs = [
                    eq._replace(k=[
                        _[2] & eq.b for _ in self.mask
                    ]) for eq in self.eqs
                ]
            else:
                assert type(mask) is list and len(mask) == self.n
                self.mask = [getMask(_) for _ in mask]
                self.eqs = [
                    eq._replace(k=[
                        _[2] & eq.b for _ in self.mask[i]
                    ]) for i, eq in enumerate(self.eqs)
                ]


if __name__ == '__main__':
    hnp = HNP(32, 3, mask='31 # 20  10 # 5', prime_full=True)
    for each in hnp.eqs:
        print([bin(_)[2:] for _ in each.k])
        print(bin(each.b)[2:])
