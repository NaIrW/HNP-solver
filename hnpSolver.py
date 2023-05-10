from hnp import HNP
from g6k import Siever
from g6k.siever_params import SieverParams
from g6k.algorithms.workout import workout
from g6k.utils.stats import SieveTreeTracer
from g6k.siever import SaturationError
from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from gmpy2 import invert
from fpylll import LLL, BKZ, IntegerMatrix, GSO
import random
from tqdm import trange
from time import time
from multiprocessing import Process, Manager, Lock


DEBUG = True

manager = Manager()
solution = manager.Value(list, [])
lock = Lock()


def shuffle(G):
    M = [[_ for _ in row] for row in G]
    random.shuffle(M)
    return M


class Solver:

    def genLattice(self):

        A = [_.a for _ in self.eqs]
        B = [_.k for _ in self.eqs]

        w = 2 ** (self.miu - 1)

        A_ = [(a * int(invert(A[0], hnp.q))) % hnp.q for a in A]
        B_ = [(b - a * B[0] + w  - w * a) % hnp.q for a, b in zip(A_, B)]

        M = [[0 for _ in range(self.n + 1)] for _ in range(self.n + 1)]

        M[0] = A_[1:] + [1, 0]
        M[1] = B_[1:] + [0, w]

        for i in range(2, self.n + 1):
            M[i] = [0] * (i - 2) + [hnp.q] + [0] * (self.n - i) + [0, 0]
        M = IntegerMatrix.from_matrix(M)
        M = GSO.Mat(
                M,
                U=IntegerMatrix.identity(M.nrows, int_type=M.int_type),
                UinvT=IntegerMatrix.identity(M.nrows, int_type=M.int_type),
                flags=GSO.ROW_EXPO,
            )
        M.update_gso()
        return M

    def predicate(self, vector):
        
        if vector[-1] == self.cw:
            x = ((self.cK + self.cw - vector[-2]) * invert(self.cA , self.q)) % self.q
        elif vector[-1] == self.ncw:
            x = ((self.cK + self.cw + vector[-2]) * invert(self.cA , self.q)) % self.q
        else:
            return False
        
        for a, b in zip(self.A, self.B):  # check high
            if ((a * x) % self.q) >> self.miu << self.miu != b:  
                return False
        else:
            return int(x)

    def lllPrep(self):
        if DEBUG:
            s_time = time()
            print('[+] LLL')

        LLL.Reduction(self.M)()

        if DEBUG:
            print(f'[+] LLL done  cost: {time() -  s_time} s')
    
    def bkzPrep(self):
        max_loops = 8

        block_size = self.M.d - 20
        
        params = SieverParams(threads=self.threads)
        g6k = Siever(self.M, params)
        tracer = SieveTreeTracer(g6k, root_label="bkz-sieve")

        if DEBUG:
            s_time = time()
            print('[+] bkz jump 20 to M.d-20')
        
        for b in range(20, block_size + 1, 10):
            pump_n_jump_bkz_tour(g6k, tracer, b, pump_params={"down_sieve": True})

        if DEBUG:
            print(f'[+] bkz jump done  cost: {time() -  s_time} s')

        auto_abort = BKZ.AutoAbort(self.M, self.M.d)

        if DEBUG:
            s_time = time()
            print('[+] bkz loop')

        for _ in range(max_loops):
            pump_n_jump_bkz_tour(g6k, tracer, block_size, pump_params={"down_sieve": True})

            if auto_abort.test_abort():
                break

            for v in self.M.B:
                if self.predicate(v):
                    self.solution.append(self.predicate(v))
        
        if DEBUG:
            print(f'[+] bkz loop done  cost: {time() -  s_time} s')
    
    def sieve(self):

        params = SieverParams(reserved_n=self.M.d, otf_lift=False, threads=self.threads)
        g6k = Siever(self.M, params)

        tracer = SieveTreeTracer(g6k, root_label="sieve")

        if DEBUG:
            s_time = time()
            print('[+] sieve')

        workout(g6k, tracer, 0, self.M.d, dim4free_min=0, dim4free_dec=15)

        if DEBUG:
            print(f'[+] sieve done  cost: {time() -  s_time} s')

        for i in range(g6k.M.d):
            if self.predicate(g6k.M.B[i]):
                self.solution.append(self.predicate(g6k.M.B[i]))
        self.g6k = g6k
    
    def findL2R(self, left, right):

        for i in range(left, min(self.db_size, right)):
            
            v = self.g6k.M.B.multiply_left(self.db[i])
            if self.predicate(v):
                lock.acquire()
                solution.value = solution.value + [self.predicate(v)]
                lock.release()
        
    def enumDatabase(self):
        
        g6k = self.g6k
        try:
            g6k()
        except SaturationError:
            pass

        while g6k.l:
            g6k.extend_left()
            try:
                g6k()
            except SaturationError:
                pass

        if DEBUG:
            s_time = time()
            print('[+] fill database')

        with g6k.temp_params(saturation_ratio=0.7, db_size_factor=3.5):
            g6k()

        if DEBUG:
            print(f'[+] fill database done  cost: {time() -  s_time} s')

        for i in range(g6k.M.d):
            if self.predicate(g6k.M.B[i]):
                self.solution.append(self.predicate(g6k.M.B[i]))
        self.db_size = g6k.db_size()
        
        if DEBUG:
            s_time = time()
            print(f'[+] checking from the database with size {self.db_size}')

        self.g6k = g6k
        self.db = list(g6k.itervalues())
        process_list = []
        gap = g6k.db_size() // self.threads

        for i in range(self.threads):
            left, right = i * gap, (self.db_size if i == self.threads - 1 else (i + 1) * gap)
            process = Process(target=self.findL2R, args=((left, right)))
            process_list.append(process)

        for t in process_list:
            t.start()

        for t in process_list:
            t.join()

        self.solution = self.solution + solution.value
        solution.value = []
        
        if DEBUG:
            print(f'[+] checking done  cost: {time() -  s_time} s')

    def __init__(self, hnp: HNP, threads=1):

        self.threads = threads

        self.hnp = hnp
        self.n = hnp.n
        self.m = hnp.m
        self.q = hnp.q
        self.miu = hnp.m - hnp.known
        self.eqs = hnp.eqs

        self.A = [_.a for _ in hnp.eqs]
        self.B = [_.k for _ in hnp.eqs]

        self.cA = self.eqs[0].a
        self.cK = self.eqs[0].k
        self.cw = 2 ** (self.miu - 1)
        self.ncw = -2 ** (self.miu - 1)

        self.M = self.genLattice()
    
    def __call__(self, **kwds):

        self.solution = []

        self.lllPrep()

        self.bkzPrep()

        self.sieve()

        self.enumDatabase()

        return self.solution


if __name__ == '__main__':
    
    res = 0
    for i in trange(10):
        hnp = HNP(256, 85, 3, prime_full=True)

        solver = Solver(hnp, threads=24)
        result = solver()
        print(f'got {len(result)}')
        if hnp.x in result:
            print(f'find hidden number!')
            res += 1
    print(res)
