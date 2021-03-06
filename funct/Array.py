import funct.lazy as L

import itertools
import math
import multiprocessing
import operator
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import reduce


class Array(list, L.ASeq):
    """
    A functional mutable sequence
    inheriting from the built-in list.
    """

    __slots__ = []
    __baseIterables = (list, range, tuple)

    def __init__(self, *args):
        """
        Constructs an Array from arguments.
        Nested lists, tuples and range objects are converted to
        Arrays, while other iterables will stored as Array elements.
        """
        if len(args) == 1 and isinstance(args[0], Iterable):
            args = list(args[0])

        if any(map(lambda e: isinstance(e, Array.__baseIterables), args)):
            super().__init__(self.__convert(a) for a in args)
        else:
            super().__init__(args)

    def add(self, e, inplace=False):
        """
        Element-wise addition with given scalar or sequence.
        """
        return self.__operate(operator.add, e, inplace)

    def sub(self, e, inplace=False):
        """
        Element-wise subtraction with given scalar or sequence.
        """
        return self.__operate(operator.sub, e, inplace)

    def mul(self, e, inplace=False):
        """
        Element-wise multiplication with given scalar or sequence.
        """
        return self.__operate(operator.mul, e, inplace)

    def div(self, e, inplace=False):
        """
        Element-wise division with given scalar or sequence.
        """
        return self.__operate(operator.truediv, e, inplace)

    def pow(self, e, inplace=False):
        """
        Raises elements of this Array to given power,
        or sequence of powers, element-wise.
        """
        return self.__operate(operator.pow, e, inplace)

    def mod(self, e, inplace=False):
        """
        Computes the remainder between elements in this Array
        and given scalar or sequence, element-wise.
        """
        return self.__operate(operator.mod, e, inplace)

    def bitwise_and(self, e, inplace=False):
        """
        Computes the bit-wise AND between elements in this Array
        and given scalar or sequence, element-wise.
        """
        return self.__operate(operator.and_, e, inplace)

    def bitwise_or(self, e, inplace=False):
        """
        Computes the bit-wise OR between elements in this Array
        and given scalar or sequence, element-wise.
        """
        return self.__operate(operator.or_, e, inplace)

    def abs(self, inplace=False):
        """ Element-wise absolute value. """
        a = self.__map(abs)
        if inplace:
            return self.__setinplace(a, inplace)
        return Array(a)

    def sum(self, start=0):
        """ Returns the sum of the Array elements. """
        if self.__isnd():
            return reduce(
                lambda a, b: sum(b, a) if isinstance(b, Array) else a + b, self, start
            )
        return sum(self, start)

    def product(self, start=1):
        """ Returns the product of the Array elements. """
        if self.__isnd():
            return reduce(
                lambda a, b: (a * b.product()) if isinstance(b, Array) else a * b,
                self,
                start,
            )
        return reduce(lambda a, b: a * b, self, start)

    def mean(self):
        """ Returns the average of the Array elements. """
        if not self.__isnd():
            return sum(self) / self.size
        else:
            return self.sum() / self.numel

    def average(self, weights=None):
        """ Returns the weighted average of the Array elements. """
        if weights is None:
            return self.mean()
        else:
            if self.__isnd():
                raise NotImplementedError(
                    "Weigthed average not supported for multidimensional Arrays"
                )
            self.__validate_seq(weights)
            return self.mul(weights).sum() / sum(weights)

    def floor(self, inplace=False):
        """ Floors the Array elements. """
        a = self.__map(math.floor)
        if inplace:
            return self.__setinplace(a, inplace)
        return Array(a)

    def ceil(self, inplace=False):
        """ Ceils the Array elements. """
        a = self.__map(math.ceil)
        if inplace:
            return self.__setinplace(a, inplace)
        return Array(a)

    def round(self, d=0, inplace=False):
        """ Rounds the Array to the given number of decimals. """
        a = self.__map(lambda e: round(e, d))
        if inplace:
            return self.__setinplace(a, inplace)
        return Array(a)

    def gt(self, e, inplace=False):
        """ Returns x > y element-wise """
        return self.__operate(operator.gt, e, inplace)

    def ge(self, e, inplace=False):
        """ Returns x >= y element-wise """
        return self.__operate(operator.ge, e, inplace)

    def lt(self, e, inplace=False):
        """ Returns x < y element-wise """
        return self.__operate(operator.lt, e, inplace)

    def le(self, e, inplace=False):
        """ Returns x <= y element-wise """
        return self.__operate(operator.le, e, inplace)

    def eq(self, e, inplace=False):
        """ Returns x == y element-wise """
        return self.__operate(operator.eq, e, inplace)

    def ne(self, e, inplace=False):
        """ Returns x != y element-wise """
        return self.__operate(operator.ne, e, inplace)

    def accumulate(self, func=operator.add):
        """
        Returns accumulated Array of elements using provided function.
        Defaults to accumulated sum.
        """
        return Array(itertools.accumulate(self, func))

    def clip(self, _min, _max, inplace=False):
        """
        Clip the values in the Array between the interval (`_min`, `_max`).
        """
        a = self.__map(lambda e: max(min(e, _max), _min))
        if inplace:
            return self.__setinplace(a, inplace)
        return Array(a)

    def roll(self, n, inplace=False):
        """ Rolls the elements of the Array. """
        n = n % self.size
        a = self[-n:] + self[:-n]
        if inplace:
            return self.__setinplace(a, inplace)
        return a

    def cat(self, b):
        """ Concatenates a sequence to this Array. """
        return self + Array(b)

    def diff(self, n=1):
        """ Returns the n-th discrete difference of the Array. """
        if n == 1:
            return self[1:].sub(self[:-1])
        else:
            return self[1:].sub(self[:-1]).diff(n - 1)

    def difference(self, b):
        """
        Difference between this Array and another iterable.
        Returns the values in this Array that are not in sequence b.
        """
        s = set(b)
        return Array(e for e in self if e not in s)

    def setdifference(self, b):
        """
        Difference between this Array and another iterable.
        Returns the unique values in this Array that are not in sequence b.
        Does not preserve order of elements.
        """
        return Array(set(self).difference(set(b)))

    def intersect(self, b):
        """
        Intersection between this Array and another iterable.
        Returns the values that are both in this Array and sequence b.
        """
        s = set(b)
        return Array(e for e in self if e in s)

    def setintersect(self, b):
        """
        Intersection between this Array and another iterable.
        Returns the unique values that are both in this Array and sequence b.
        Does not preserve order of elements.
        """
        return Array(set(self).intersection(set(b)))

    def union(self, b):
        """
        Union of this Array and another iterable.
        Returns the values that are in either this Array or sequence b.
        """
        return self + b

    def setunion(self, b):
        """
        Union of this Array and another iterable.
        Returns the unique values that are in either this Array or sequence b.
        Does not preserve order of elements.
        """
        return Array(set(self).union(set(b)))

    def unique(self):
        """
        Selects unique values in this Array.
        Does not preserve order of elements.
        """
        return Array(set(self))

    def nunique(self):
        """ Returns the number of unique elements in the Array. """
        return len(set(self))

    def equal(self, b):
        """ Returns true this Array and given sequence have the same elements. """
        if len(self) != len(b):
            return False
        if any(map(lambda e: isinstance(e, Array), self)):
            try:
                return all(
                    map(
                        lambda x, y: x.equal(y) if isinstance(x, Array) else x == y,
                        self,
                        b,
                    )
                )
            except TypeError:
                return False
        return all(self.eq(b))

    def remove(self, b, inplace=False):
        """ Removes first occurence(s) of the value(s). """
        a = self if inplace else self.copy()
        if isinstance(b, Iterable):
            for i in b:
                super(Array, a).remove(i)
        else:
            super(Array, a).remove(b)
        return a

    def remove_by_index(self, b, inplace=False):
        """ Removes the value at specified index or indices. """
        a = self if inplace else self.copy()
        if isinstance(b, Iterable):
            try:
                c = 0
                for i in b:
                    a.pop(i - c)
                    c += 1
            except TypeError:
                raise TypeError(
                    "{} object cannot be interpreted as an integer".format(
                        type(i).__name__
                    )
                ) from None
        else:
            try:
                a.pop(b)
            except TypeError:
                raise TypeError(
                    "{} object cannot be interpreted as an integer".format(
                        type(b).__name__
                    )
                ) from None
        return a

    def get(self, key, default=None):
        """
        Safe method for getting elements in the Array.
        Returns default if the index does not exist.
        """
        try:
            return self[key]
        except IndexError:
            return default

    def map(self, func, inplace=False):
        """
        Returns an Array by applying provided function to each element in this Array.
        """
        a = map(func, self)
        if inplace:
            return self.__setinplace(a, inplace)
        return Array(a)

    def apply(self, func, inplace=False):
        """
        Applies function to all non-Array elements of the Array.
        """
        a = self.__map(func)
        if inplace:
            return self.__setinplace(a, inplace)
        return Array(a)

    def starmap(self, func):
        """
        Returns an Array by applying provided function
        to this Array with itertools.starmap
        """
        return Array(itertools.starmap(func, self))

    def parmap(self, func, processes=None, Pool=multiprocessing.Pool):
        """
        Returns an Array by applying a function to
        all elements of this Array in parallel.

        Parameters
        ----------
        func: Callable
        processes: int, optional
        Pool: Pool, optional (multiprocessing.Pool, multiprocessing.pool.ThreadPool)
        """
        with Pool(processes=processes) as pool:
            return Array(pool.map(func, self))

    def parstarmap(self, func, processes=None, Pool=multiprocessing.Pool):
        """
        Parallel starmap

        Parameters
        ----------
        func: Callable
        processes: int, optional
        Pool: Pool, optional (multiprocessing.Pool, multiprocessing.pool.ThreadPool)
        """
        with Pool(processes=processes) as pool:
            return Array(pool.starmap(func, self))

    def asyncmap(self, func, Executor=ThreadPoolExecutor):
        """
        Executes map asynchronously.
        Returns a Future object.

        Parameters
        ----------
        func: Callable
        Executor: Executor, optional (concurrent.futures.ThreadPoolExecutor, concurrent.futures.ProcessPoolExecutor)
        """
        with Executor() as executor:
            return executor.submit(self.map, func)

    def asyncstarmap(self, func, Executor=ThreadPoolExecutor):
        """
        Executes starmap asynchronously.
        Returns a Future object.

        Parameters
        ----------
        func: Callable
        Executor: Executor, optional (concurrent.futures.ThreadPoolExecutor, concurrent.futures.ProcessPoolExecutor)
        """
        with Executor() as executor:
            return executor.submit(self.starmap, func)

    def filter(self, func):
        """ Selects elements of this Array which satisfy the predicate. """
        return Array(filter(func, self))

    def forall(self, func):
        """
        Returns whether the specified predicate
        holds for all elements of this Array.
        """
        return all(map(func, self))

    def forany(self, func):
        """
        Returns whether the specified predicate
        holds for any element of this Array.
        """
        return any(map(func, self))

    def foreach(self, func):
        """ Applies `func` to each element of the Array. """
        for e in self:
            func(e)

    def reduce(self, func, init=None):
        """ Reduces the elements of this Array using the specified operator. """
        if init is not None:
            return reduce(func, self, init)
        return reduce(func, self)

    def contains(self, e):
        """ Tests whether element exists in this Array. """
        return e in self

    def index_where(self, func):
        """ Finds the index of the first element satisfying a predicate. """
        for i, v in enumerate(self):
            if func(v):
                return i
        raise ValueError("No matches")

    def indices_where(self, func):
        """ Finds all the indices of the elements satisfying a predicate. """
        return Array(i for i, v in enumerate(self) if func(v))

    def indices(self, e):
        """ Returns all the indices of provided value in this Array. """
        r = []
        o = -1
        while True:
            try:
                o = self.index(e, o + 1)
            except ValueError:
                return Array(r)
            r.append(o)

    def split(self, c):
        """
        Splits an Array into subarrays using provided argument as the delimiter element.
        """
        try:
            i = self.index(c)
            v = self[:i].unsqueeze if i != 0 else Array()
            for e in self[i + 1 :].split(c):
                if len(e) != 0:
                    v.append(e)
            return v
        except ValueError:
            return self.unsqueeze

    def split_to(self, n):
        """ Splits this Array to n equal length subarrays """
        if self.size % n != 0:
            raise ValueError("Split does not result in an equal division")
        d = self.size // n
        return Array(self[d * i : d * (i + 1)] for i in range(n))

    def split_at(self, n):
        """ Splits this Array into subarrays at specified index or indices. """
        if isinstance(n, int):
            return Array(self[:n], self[n:])
        n = Array(0, *n, self.size)
        return Array(self[n[i] : n[i + 1]] for i in range(n.size - 1))

    def chunks(self, n, drop_last=False):
        """
        Splits this Array into chunks of size n.
        If `drop_last` is True, drops the last subarray if the split
        results in an inequal division.
        """
        self.__validate_bool_arg(drop_last, "drop_last")
        fun = int if drop_last else math.ceil
        return Array(self[i * n : (i + 1) * n] for i in range(fun(self.size / n)))

    def windows(self, size, stride=1, drop_last=True):
        """
        Returns sliding windows of width `size` over the Array.
        If `drop_last` is True, drops the last subarrays that are
        shorter than the window size.
        """
        self.__validate_bool_arg(drop_last, "drop_last")
        end = (self.size - size + 1) if drop_last else self.size
        return Array(self[i : i + size] for i in range(0, end, stride))

    def takewhile(self, func):
        """ Takes the longest prefix of elements that satisfy the given predicate. """
        return Array(itertools.takewhile(func, self))

    def dropwhile(self, func):
        """ Drops the longest prefix of elements that satisfy the given predicate. """
        return Array(itertools.dropwhile(func, self))

    def groupby(self, func):
        """
        Groups this Array into an Array of Array-tuples according
        to given discriminator function.
        """
        return Array(
            (e, Array(g)) for e, g in itertools.groupby(sorted(self, key=func), func)
        )

    def maxby(self, func, **kwargs):
        """ Finds the maximum value measured by a function. """
        return max(self, key=func, **kwargs)

    def minby(self, func, **kwargs):
        """ Finds the minimum value measured by a function. """
        return min(self, key=func, **kwargs)

    def sortby(self, func, reverse=False, inplace=False):
        """
        Sorts this Array according to a function
        defining the sorting criteria.
        """
        if inplace:
            self.__validate_bool_arg(inplace, "inplace")
            super().sort(key=func, reverse=reverse)
            return self
        return Array(sorted(self, key=func, reverse=reverse))

    def argsortby(self, func, reverse=False):
        """
        Returns the indices that would sort this Array according to
        provided sorting criteria.
        """
        return self.enumerate.sortby(lambda e: func(e[1]), reverse=reverse)[:, 0]

    def sort(self, inplace=False, **kwargs):
        """ Sorts this Array. """
        if inplace:
            self.__validate_bool_arg(inplace, "inplace")
            super().sort(**kwargs)
            return self
        return Array(sorted(self, **kwargs))

    def argsort(self, reverse=False):
        """ Returns the indices that would sort this Array """
        return self.enumerate.sortby(lambda e: e[1], reverse=reverse)[:, 0]

    def reverse(self, inplace=False):
        """ Reverses this Array. """
        if inplace:
            self.__validate_bool_arg(inplace, "inplace")
            super().reverse()
            return self
        return Array(reversed(self))

    def copy(self):
        return Array(super().copy())

    def astype(self, t):
        """
        Converts the elements in this Array to given type.
        """
        return Array(map(t, self))

    def join(self, delimiter=" "):
        """
        Creates a string representation of this Array
        with elements separated with `delimiter`
        """
        return delimiter.join(str(v) for v in self)

    def append(self, e):
        """
        Appends an element to the end of this Array.
        """
        super().append(self.__convert(e))
        return self

    def prepend(self, e):
        """
        Prepends an element to the beginning of this Array.
        """
        super().insert(0, self.__convert(e))
        return self

    def extend(self, e):
        """ Extend this Array by appending elements from the iterable. """
        super().extend(Array(e))
        return self

    def extendleft(self, e):
        """ Extends this Array by prepending elements from the iterable. """
        self[0:0] = Array(e)
        return self

    def insert(self, i, e):
        """ Inserts element(s) (in place) before given index/indices. """
        if isinstance(e, Array.__baseIterables) and isinstance(
            i, Array.__baseIterables
        ):
            if len(e) != len(i):
                raise ValueError(
                    "The lengths of the sequences must match, got {} and {}".format(
                        len(i), len(e)
                    )
                )
            for ii, ei in zip(i, e):
                self.insert(ii, ei)
        else:
            super().insert(i, self.__convert(e))
        return self

    def fill(self, e, inplace=False):
        """ Replaces all elements of this Array with given object. """
        if inplace:
            return self.__setinplace(e, inplace)
        return Array([e] * self.size)

    def pad(self, n, value=0):
        """ Pads this Array with value. """
        try:
            return self + [value] * n
        except TypeError:
            raise TypeError(
                "{} object cannot be interpreted as an integer".format(type(n).__name__)
            ) from None

    def padleft(self, n, value=0):
        """ Pads this Array with value. """
        try:
            return [value] * n + self
        except TypeError:
            raise TypeError(
                "{} object cannot be interpreted as an integer".format(type(n).__name__)
            ) from None

    def pad_to(self, n, value=0):
        """
        Pads this Array with value until length of n is reached.
        """
        try:
            return self + [value] * (n - self.size)
        except TypeError:
            raise TypeError(
                "{} object cannot be interpreted as an integer".format(type(n).__name__)
            ) from None

    def padleft_to(self, n, value=0):
        """ Pads this Array with value until length of n is reached. """
        try:
            return [value] * (n - self.size) + self
        except TypeError:
            raise TypeError(
                "{} object cannot be interpreted as an integer".format(type(n).__name__)
            ) from None

    def zip(self, *args):
        return Array(zip(self, *args))

    def unzip(self):
        """
        'Unzips' nested Arrays by unpacking its elements into a zip.

        >>> Array((1, "a"), (2, "b")).unzip()
        Array(Array(1, 2), Array('a', 'b'))
        """
        if not all(map(lambda e: isinstance(e, Array), self)):
            raise TypeError("Array elements must support iteration")
        return Array(zip(*self))

    def zip_all(self, *args, default=None):
        """
        Zips the sequences. If the iterables are
        of uneven length, missing values are filled with default value.
        """
        return Array(itertools.zip_longest(self, *args, fillvalue=default))

    def all(self):
        """ Returns true if bool(e) is True for all elements in this Array. """
        return all(self)

    def any(self):
        """ Returns true if bool(e) is True for any element e in this Array. """
        return any(self)

    def max(self, **kwargs):
        return max(self, **kwargs)

    def argmax(self):
        """ Returns the index of the maximum value """
        return max(self.enumerate, key=lambda e: e[1])[0]

    def min(self, **kwargs):
        return min(self, **kwargs)

    def argmin(self):
        """ Returns the index of the minimum value """
        return min(self.enumerate, key=lambda e: e[1])[0]

    @property
    def head(self):
        """ Selects the first element of this Array. """
        return self[0]

    def head_option(self, default=None):
        """
        Selects the first element of this Array if it has one,
        otherwise returns default.
        """
        return self.get(0, default)

    @property
    def last(self):
        """ Selects the last element of this Array. """
        return self[-1]

    def last_option(self, default=None):
        """
        Selects the last element of this Array if it has one,
        otherwise returns default.
        """
        return self.get(-1, default)

    @property
    def init(self):
        """ Selects the rest of this Array without its last element. """
        return self[:-1]

    @property
    def tail(self):
        """ Selects the rest of this Array without its first element. """
        return self[1:]

    @property
    def isempty(self):
        """ Returns whether this Array is empty. """
        return self.size == 0

    @property
    def nonempty(self):
        """ Returns whether this Array is not empty. """
        return self.size != 0

    def isfinite(self):
        """
        Tests element-wise whether the elements are neither infinity nor NaN.
        """
        return Array(self.__map(math.isfinite))

    @property
    def length(self):
        """ Length of the Array. """
        return self.__len__()

    @property
    def numel(self):
        """ Total number of elements in the Array. """
        if not self.__isnd():
            return len(self)
        return sum(e.numel if isinstance(e, Array) else 1 for e in self)

    @property
    def size(self):
        """ Length of the Array. """
        return self.__len__()

    def int(self):
        """ Converts elements in this Array to integers. """
        try:
            return Array(self.__map(lambda e: ord(e) if isinstance(e, str) else int(e)))
        except TypeError:
            raise TypeError("Expected an Array of numbers or characters") from None

    def float(self):
        """ Converts elements in this Array to floats. """
        return Array(self.__map(float))

    def bool(self):
        """ Converts elements in this Array to booleans. """
        return Array(self.__map(bool))

    def char(self):
        """ Converts an Array of integers to chars. """
        return Array(self.__map(chr))

    def to_str(self):
        """
        Creates a string representation of the Array.
        See Array.join for more functionality.
        """
        return "".join(str(v) for v in self)

    def to_tuple(self):
        """ Returns a copy of the Array as an tuple. """
        return tuple(self.copy())

    def to_set(self):
        """ Returns set of the Array elements. Order is not preserved. """
        return set(self)

    def to_iter(self):
        """ Returns an iterator for the Array."""
        for e in self:
            yield e

    def to_dict(self):
        """ Creates a dictionary from an Array of Arrays / tuples. """
        return dict(self)

    @property
    def squeeze(self):
        """
        Returns the Array with the same elements, but with
        outermost singleton dimension removed (if exists).
        """
        if isinstance(self.head_option(), Array) and self.length == 1:
            return self[0]
        else:
            return self

    @property
    def unsqueeze(self):
        """
        Returns the Array with the same elements wrapped in a Array.
        """
        return Array([self])

    @property
    def flatten(self):
        """ Returns the Array collapsed into one dimension. """
        r = Array(e for s in self for e in (s if isinstance(s, Array) else [s]))
        if any(map(lambda e: isinstance(e, Array), r)):
            return r.flatten
        return r

    @property
    def range(self):
        """ Returns the indices of the Array"""
        return Array(range(self.size))

    @property
    def enumerate(self):
        """ Zips the Array with its indices """
        return Array(enumerate(self))

    @staticmethod
    def zeros(n):
        """ Returns a zero-filled Array of given length. """
        try:
            return Array([0] * n)
        except TypeError:
            raise TypeError(
                "{} object cannot be interpreted as an integer".format(type(n).__name__)
            ) from None

    @staticmethod
    def zeros_like(e):
        """ Returns a zero-filled Array with same shape as given Array. """
        if isinstance(e, Array.__baseIterables):
            return Array(e).mul(0)
        raise TypeError(
            "Expected one of ("
            + ", ".join(str(e) for e in Array.__baseIterables)
            + f") got {type(e)}"
        )

    @staticmethod
    def arange(*args):
        """
        Returns an Array of evenly spaced values within given interval

        Parameters
        ----------
        start: number, optional
        end: number
        step: number, optional
        """
        _len = len(args)
        if _len == 0:
            raise TypeError("Expected at least one argument")
        elif _len > 3:
            raise TypeError("Expected at most 3 arguments")
        elif _len == 1:
            return Array(range(math.ceil(args[0])))
        start, end, step = args if _len == 3 else args + (1,)
        return Array(start + step * i for i in range(math.ceil((end - start) / step)))

    @staticmethod
    def linspace(start, stop, num=50, endpoint=True):
        """
        Returns evenly spaced Array over a specified interval.

        Parameters
        ----------
        start: number
        stop: number
        num: int, optional
        endpoint : bool, optional
        """
        Array.__validate_bool_arg(None, endpoint, "endpoint")
        step = (stop - start) / max(num - bool(endpoint), 1)
        return Array(start + step * i for i in range(num))

    @staticmethod
    def logspace(start, stop, num=50, base=10, endpoint=True):
        """
        Returns Array spaced evenly on a log scale.

        Parameters
        ----------
        start: number
        stop: number
        num: int, optional
        base: float, optional
        endpoint: bool, optional
        """
        Array.__validate_bool_arg(None, endpoint, "endpoint")
        return base ** Array.linspace(start, stop, num, endpoint)

    def __convert(self, e):
        return Array(e) if isinstance(e, Array.__baseIterables) else e

    def __validate_index(self, i):
        if not isinstance(i, (int, slice)):
            if not isinstance(i, Sequence) or not all(
                map(lambda e: isinstance(e, (bool, int, slice)), i)
            ):
                raise TypeError(
                    "Only integers, slices and 1d integer or boolean Arrays are valid indices"
                )

    def __validate_setelem(self, i, e):
        if isinstance(e, Iterable):
            if len(i) != len(e):
                raise ValueError(
                    "Expected Array of size {}, got {}".format(len(i), len(e))
                )
            return iter(e)
        else:
            return itertools.repeat(e)

    def __validate_seq(self, e):
        if len(e) != self.size:
            raise ValueError(
                "The lengths of the sequences must match, got {} and {}".format(
                    self.size, len(e)
                )
            )

    def __validate_bool_arg(self, value, name):
        if not isinstance(value, bool):
            raise ValueError(f"Expected type bool for {name} argument")

    def __setinplace(self, s, arg):
        if not isinstance(arg, bool):
            raise ValueError("Expected type bool for inplace argument")
        self[:] = s
        return self

    def __isnd(self):
        return any(map(lambda e: isinstance(e, Array), self))

    def __map(self, f):
        if any(map(lambda e: isinstance(e, Array), self)):
            f = lambda x, f=f: (Array(x.__map(f)) if isinstance(x, Array) else f(x))
        return map(f, self)

    def __map2(self, f, e):
        if any(map(lambda e: isinstance(e, Array), self)):
            f = lambda x, y, f=f: (
                Array(x.__map2(f, y)) if isinstance(x, Array) else f(x, y)
            )
        if isinstance(e, Iterable):
            return map(f, self, e)
        return map(f, self, itertools.repeat(e))

    def __operate(self, op, e, inplace):
        if not isinstance(inplace, bool):
            raise ValueError("Expected type bool for inplace argument")
        a = self.__map2(op, e)
        if inplace:
            self[:] = a
            return self
        return Array(a)

    def __repr__(self):
        return "Array" + "(" + super().__repr__()[1:-1] + ")"

    def __add__(self, b):
        if isinstance(b, Array.__baseIterables):
            return Array(super().__add__(Array(b)))
        else:
            raise TypeError(f"Can not concatenate {type(b).__name__} to Array")

    def __iadd__(self, b):
        if isinstance(b, Array.__baseIterables):
            super().__iadd__(b)
            return self
        else:
            raise TypeError(f"Can not concatenate {type(b).__name__} to Array")

    def __radd__(self, b):
        if isinstance(b, Array.__baseIterables):
            return Array(b) + self
        else:
            return b + sum(self)

    def __pow__(self, b):
        return self.pow(b)

    def __rpow__(self, b):
        if isinstance(b, Array.__baseIterables):
            return Array(b).pow(self)
        return Array([b] * self.size).pow(self)

    def __gt__(self, e):
        return self.__operate(operator.gt, e, False)

    def __ge__(self, e):
        return self.__operate(operator.ge, e, False)

    def __lt__(self, e):
        return self.__operate(operator.lt, e, False)

    def __le__(self, e):
        return self.__operate(operator.le, e, False)

    def __eq__(self, e):
        return self.__operate(operator.eq, e, False)

    def __ne__(self, e):
        return self.__operate(operator.ne, e, False)

    def __and__(self, e):
        return self.__operate(operator.and_, e, False)

    def __or__(self, e):
        return self.__operate(operator.or_, e, False)

    def __neg__(self):
        return Array(map(operator.neg, self))

    def __invert__(self):
        if all(map(lambda e: isinstance(e, bool), self)):
            return Array(map(lambda e: not e, self))
        return Array(map(operator.inv, self))

    def __hash__(self):
        return hash(self.to_tuple())

    def __bool__(self):
        if self.__len__() == 0:
            warnings.warn(
                "The truth value of an empty Array is ambiguous. Use `Array.nonEmpty` to check that Array is not empty",
                UserWarning,
            )
        return all(self)

    def __setitem__(self, key, e):
        self.__validate_index(key)
        if isinstance(key, tuple):
            if len(key) == 1:
                key = key[0]
            else:
                if isinstance(key[0], int):
                    self[key[0]][key[1:]] = e
                    return
                try:
                    idx = range(*key[0].indices(self.size))
                    for ic in idx:
                        self[ic][key[1:]]
                    for ic, ie in zip(idx, self.__validate_setelem(idx, e)):
                        self[ic][key[1:]] = ie
                except TypeError:
                    raise IndexError("Too many indices for the Array") from None
                return
        if isinstance(key, int):
            super().__setitem__(key, self.__convert(e))
            return
        if isinstance(key, Sequence):
            for _i, _e in zip(key, self.__validate_setelem(key, e)):
                super().__setitem__(_i, _e)
            return
        if isinstance(e, Iterable):
            e = self.__convert(e)
        else:
            e = [e] * len(range(*key.indices(self.size)))
        super().__setitem__(key, e)

    def __getitem__(self, key):
        self.__validate_index(key)
        if isinstance(key, tuple):
            if len(key) == 1:
                key = key[0]
            else:
                if isinstance(key[0], int):
                    return self[key[0]][key[1:]]
                try:
                    return Array(e[key[1:]] for e in super().__getitem__(key[0]))
                except TypeError:
                    raise IndexError("Too many indices for the Array") from None
        if isinstance(key, int):
            return super().__getitem__(key)
        if isinstance(key, slice):
            return Array(super().__getitem__(key))
        if all(map(lambda k: isinstance(k, bool), key)):
            if len(key) != self.size:
                raise IndexError(
                    "Expected boolean Array of size {}, got {}".format(
                        self.size, len(key)
                    )
                )
            key = (i for i, k in enumerate(key) if k)
        return Array(map(super().__getitem__, key))
