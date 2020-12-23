import math
import multiprocessing
import operator
from collections.abc import Iterable
from functools import reduce
from itertools import repeat, starmap, zip_longest
from numbers import Number
from operator import add, mul, pow, gt, ge, lt, le, eq, ne, mod, and_, or_, inv


class Array(list):
    """
    A functional mutable sequence
    inheriting from the built-in list.
    """

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

    def abs(self):
        """ Element-wise absolute value. """
        return self.map(abs)

    def abs_(self):
        """ Inplace element-wise absolute value. """
        self[:] = self.abs
        return self

    def add(self, e):
        """
        Element-wise addition with given scalar or sequence.
        """
        return self.__operate(add, e)

    def add_(self, e):
        """
        Inplace element-wise addition with given scalar or sequence.
        """
        self[:] = self.__operate(add, e)
        return self

    def mul(self, e):
        """
        Element-wise multiplication with given scalar or sequence.
        """
        return self.__operate(mul, e)

    def mul_(self, e):
        """
        Inplace element-wise multiplication with given scalar or sequence.
        """
        self[:] = self.__operate(mul, e)
        return self

    def pow(self, e):
        """
        Raises elements of this Array to given power,
        or sequence of powers, element-wise.
        """
        return self.__operate(pow, e)

    def pow_(self, e):
        """
        Raises elements (in-place) of this Array to given power,
        or sequence of powers, element-wise.
        """
        self[:] = self.__operate(pow, e)
        return self

    def mod(self, e):
        """
        Computes the remainder between elements in this Array
        and given scalar or sequence, element-wise.
        """
        return self.__operate(mod, e)

    def mod_(self, e):
        """
        Computes (in-place) the remainder between elements in this Array
        and given scalar or sequence, element-wise.
        """
        self[:] = self.__operate(mod, e)
        return self

    def bitwiseAnd(self, e):
        """
        Computes the bit-wise AND between elements in this Array
        and given scalar or sequence, element-wise.
        """
        return self & e

    def bitwiseAnd_(self, e):
        """
        Computes (in-place) the bit-wise AND between elements in this Array
        and given scalar or sequence, element-wise.
        """
        self[:] = self & e
        return self

    def bitwiseOr(self, e):
        """
        Computes the bit-wise OR between elements in this Array
        and given scalar or sequence, element-wise.
        """
        return self | e

    def bitwiseOr_(self, e):
        """
        Computes (in-place) the bit-wise AND between elements in this Array
        and given scalar or sequence, element-wise.
        """
        self[:] = self | e
        return self

    def sum(self):
        return sum(self)

    def product(self):
        return self.reduce(lambda a, b: a * b)

    def mean(self):
        return self.sum() / self.size

    def average(self):
        return self.mean()

    def round(self, d=0):
        """ Rounds the Array to the given number of decimals. """
        return self.map(lambda e: round(e, d))

    def round_(self, d=0):
        """ Rounds the Array in-place to the given number of decimals. """
        self[:] = self.map(lambda e: round(e, d))
        return self

    def gt(self, e):
        return self.__gt__(e)

    def gt_(self, e):
        self[:] = self.__gt__(e)
        return self

    def ge(self, e):
        return self.__ge__(e)

    def ge_(self, e):
        self[:] = self.__ge__(e)
        return self

    def lt(self, e):
        return self.__lt__(e)

    def lt_(self, e):
        self[:] = self.__lt__(e)
        return self

    def le(self, e):
        return self.__le__(e)

    def le_(self, e):
        self[:] = self.__le__(e)
        return self

    def eq(self, e):
        return self.__operate(eq, e)

    def eq_(self, e):
        self[:] = self.__operate(eq, e)
        return self

    def ne(self, e):
        return self.__ne__(e)

    def ne_(self, e):
        self[:] = self.__ne__(e)
        return self

    def clip(self, _min, _max):
        """ Clip the values in the Array. """
        return self.map(lambda e: max(min(e, _max), _min))

    def clip_(self, _min, _max):
        """ Clip the values in the Array in-place. """
        self[:] = self.map(lambda e: max(min(e, _max), _min))
        return self

    def roll(self, n):
        """ Rolls the elements of the Array. """
        return self[-n:] + self[:-n]

    def roll_(self, n):
        """ Rolls the elements of the Array in-place. """
        self[:] = self[-n:] + self[:-n]
        return self

    def cat(self, b):
        """ Concatenates a sequence to this Array. """
        return self + Array(b)

    def diff(self, n=1):
        """ Returns the n-th discrete difference of the Array. """
        if n == 1:
            return self[1:].add(-self[:-1])
        else:
            return self[1:].add(-self[:-1]).diff(n - 1)

    def difference(self, b):
        """
        Difference between this Array and another iterable.
        Returns the values in this Array that are not in Array b.
        """
        s = set(b)
        return Array([e for e in self if e not in s])

    def setDifference(self, b):
        """
        Difference between this Array and another iterable.
        Returns the unique values in this Array that are not in Array b.
        Does not preserve element order.
        """
        return Array(set(self).difference(set(b)))

    def intersect(self, b):
        """
        Intersection between this Array and another iterable.
        Returns the values that are both in this Array and Array b.
        """
        s = set(b)
        return Array([e for e in self if e in s])

    def setIntersect(self, b):
        """
        Intersection between this Array and another iterable.
        Returns the unique values that are both in this Array and Array b.
        """
        return Array(set(self).intersection(set(b)))

    def union(self, b):
        """
        Union of this Array and another iterable.
        Returns the values that are in either this Array or Array b.
        """
        return self + b

    def setUnion(self, b):
        """
        Union of this Array and another iterable.
        Returns the unique values that are in either this Array or Array b.
        """
        return Array(set(self).union(set(b)))

    def equal(self, b):
        """ Returns true if Arrays have the same elements """
        return self.eq(b).all

    def remove(self, b):
        """ Removes first occurence(s) of the value(s). """
        a = self.copy()
        if isinstance(b, Iterable):
            for i in set(b):
                super(Array, a).remove(i)
        else:
            super(Array, a).remove(b)
        return a

    def remove_(self, b):
        """ Removes first occurence(s) of the value(s) in place. """
        if isinstance(b, Iterable):
            for i in set(b):
                super().remove(i)
        else:
            super().remove(b)
        return self

    def removeByIndex(self, b):
        """ Removes the value at specified index or indices. """
        a = self.copy()
        if isinstance(b, Iterable):
            try:
                c = 0
                for i in b:
                    a.pop(i - c)
                    c += 1
            except TypeError as e:
                raise TypeError(
                    "{} object cannot be interpreted as an integer".format(
                        type(i).__name__
                    )
                ) from e
        else:
            try:
                a.pop(b)
            except TypeError as e:
                raise TypeError(
                    "{} object cannot be interpreted as an integer".format(
                        type(b).__name__
                    )
                ) from e
        return a

    def removeByIndex_(self, b):
        """ Removes the value at specified index or indices in place. """
        if isinstance(b, Iterable):
            try:
                c = 0
                for i in b:
                    self.pop(i - c)
                    c += 1
            except Exception as e:
                raise TypeError(
                    "{} object cannot be interpreted as an integer".format(
                        type(i).__name__
                    )
                ) from e
        else:
            try:
                self.pop(b)
            except Exception as e:
                raise TypeError(
                    "{} object cannot be interpreted as an integer".format(
                        type(b).__name__
                    )
                ) from e
        return self

    def get(self, key, default=None):
        """
        Safe method for getting elements in the Array.
        Returns default if the index does not exist.
        """
        try:
            return self[key]
        except IndexError:
            return default

    def map(self, l):
        """
        Returns an Array by applying provided function to all elements of this Array.
        """
        return Array(map(l, self))

    def starmap(self, l):
        """
        Returns an Array by applying provided function
        to this Array with itertools.starmap
        """
        return Array(starmap(l, self))

    def parmap(self, l, n_processes=None):
        """
        Returns an Array by applying a function to
        all elements of this Array in parallel.
        """
        with multiprocessing.Pool(processes=n_processes) as pool:
            return Array(pool.map(l, self))

    def parstarmap(self, l, processes=None):
        with multiprocessing.Pool(processes=processes) as pool:
            return Array(pool.starmap(l, self))

    def filter(self, l):
        """ Selects elements of this Array which satisfy the predicate. """
        return Array(filter(l, self))

    def forall(self, l):
        """
        Returns whether the specified predicate
        holds for all elements of this Array.
        """
        return self.map(l).all

    def forany(self, l):
        """
        Returns whether the specified predicate
        holds for any elements of this Array.
        """
        return self.map(l).any

    def count(self, l):
        """ Returns the number of elements that satify the predicate """
        return self.map(l).sum

    def reduce(self, l, init=None):
        """ Reduces the elements of this Array using the specified operator. """
        if init is not None:
            return reduce(l, self, init)
        else:
            return reduce(l, self)

    def contains(self, e):
        """ Tests wheter element exists in this Array. """
        return e in self

    def indexWhere(self, l):
        """ Finds the index of the first element satisfying a predicate. """
        for i, v in enumerate(self):
            if l(v):
                return i
        raise ValueError("No matches")

    def indicesWhere(self, l):
        """ Finds all the indices of the elements satisfying a predicate. """
        return Array([i for i, v in enumerate(self) if l(v)])

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
                if not e.isEmpty:
                    v.append(e)
            return v
        except ValueError:
            return self.unsqueeze

    def splitTo(self, n):
        """ Splits this Array to n equal length subarrays """
        if self.size % n != 0:
            raise ValueError("Split does not result in an equal division")
        d = self.size // n
        return Array([self[d * i : d * (i + 1)] for i in range(n)])

    def splitAt(self, n):
        """ Splits this Array into subarrays at specified index or indices. """
        if isinstance(n, int):
            return Array(self[:n], self[n:])
        elif isinstance(n, Iterable):
            n = Array(n)
            n = Array(0, *n, self.size)
            return Array([self[n[i] : n[i + 1]] for i in range(n.size - 1)])

    def takeWhile(self, l):
        """ Takes the longest prefix of elements that satisfy the given predicate. """
        for i, v in enumerate(self):
            if not l(v):
                return self[:i]

    def dropWhile(self, l):
        """ Drops the longest prefix of elements that satisfy the given predicate. """
        for i, v in enumerate(self):
            if not l(v):
                return self[i:]

    def groupBy(self, l):
        """
        Groups this Array into a dict of Arrays according
        to given discriminator function.
        """
        m = {}
        for v in self:
            k = l(v)
            if k in m:
                m[k].append(v)
            else:
                m[k] = Array([v])
        return m

    def maxBy(self, l):
        """ Finds the maximum value measured by a function. """
        r = max(self, key=l)
        if isinstance(r, Array.__baseIterables):
            return Array(r)
        return r

    def minBy(self, l):
        """ Finds the minimum value measured by a function. """
        r = min(self, key=l)
        if isinstance(r, Array.__baseIterables):
            return Array(r)
        return r

    def sortBy(self, l):
        """
        Sorts this Array according to a function
        defining the sorting criteria.
        """
        return Array(sorted(self, key=l))

    def sortBy_(self, l):
        """
        Sorts this Array in place according to a function
        defining the sorting criteria.
        """
        super().sort(key=l)
        return self

    def argsortBy(self, l):
        """
        Returns the indices that would sort this Array according to
        provided sorting criteria.
        """
        return self.zipWithIndex.sortBy(lambda e: l(e[1])).map(lambda e: e[0])

    def sort(self, **kwargs):
        """ Sorts this Array. """
        return Array(sorted(self, **kwargs))

    def sort_(self, **kwargs):
        """ Sorts this Array in place. """
        super().sort(**kwargs)
        return self

    def argsort(self):
        """ Returns the indices that would sort this Array """
        return self.zipWithIndex.sortBy(lambda e: e[1]).map(lambda e: e[0])

    def reverse(self):
        """ Reverses this Array. """
        return Array(reversed(self))

    def reverse_(self):
        """ Reverses this Array in place. """
        super().reverse()
        return self

    def copy(self):
        return Array(super().copy())

    def asType(self, t):
        """
        Converts the elements in this Array to give type.
        """
        return Array(map(t, self))

    def mkString(self, delimiter=" "):
        """
        Creates a string representation of this Array
        with elements separated with 'delimiter'
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
        super().extend(self.__convert(e))
        return self

    def extendLeft(self, e):
        """ Extends this Array by prepending elements from the iterable. """
        for v in e:
            self.prepend(v)
        return self

    def insert(self, i, e):
        """ Inserts element(s) before given index/indices. """
        a = self.copy()
        if isinstance(e, Array.__baseIterables) and isinstance(
            i, Array.__baseIterables
        ):
            for ii, ei in zip(i, e):
                a = a.insert(ii, ei)
        else:
            super(Array, a).insert(i, e)
        return a

    def insert_(self, i, e):
        """ Inserts element(s) (in place) before given index/indices. """
        if isinstance(e, Array.__baseIterables):
            for ei in e:
                self.insert_(i, ei)
                i += 1
        else:
            super().insert(i, e)
        return self

    def fill(self, e):
        """ Replaces all elements of this Array with given object. """
        return Array([e] * self.size)

    def fill_(self, e):
        """ Replaces (in place) all elements of this Array with given object. """
        for i in range(self.size):
            self[i] = e
        return self

    def pad(self, n, value=0):
        """ Pads this Array with value. Default value=0. """
        try:
            return self + [value] * n
        except Exception as e:
            raise TypeError(
                "{} object cannot be interpreted as an integer".format(type(n).__name__)
            ) from e

    def padLeft(self, n, value=0):
        """ Pads this Array with value. Default value=0. """
        try:
            return [value] * n + self
        except Exception as e:
            raise TypeError(
                "{} object cannot be interpreted as an integer".format(type(n).__name__)
            ) from e

    def padTo(self, n, value=0):
        """
        Pads this Array with value until length of n is reached.
        Default value=0.
        """
        try:
            return self + [value] * (n - self.size)
        except Exception as e:
            raise TypeError(
                "{} object cannot be interpreted as an integer".format(type(n).__name__)
            ) from e

    def padLeftTo(self, n, value=0):
        """
        Pads this Array with value until length of n is reached.
        Default value=0.
        """
        try:
            return [value] * (n - self.size) + self
        except Exception as e:
            raise TypeError(
                "{} object cannot be interpreted as an integer".format(type(n).__name__)
            ) from e

    def zip(self, *args):
        return Array(zip(self, *args))

    def zipAll(self, *args, default=None):
        """
        Zips the sequences. If the iterables are
        of uneven length, missing values are filled with default.
        """
        return Array(zip_longest(self, *args, fillvalue=default))

    @property
    def all(self):
        """ Returns true if bool(e) is True for all elements in this Array. """
        return all(self)

    @property
    def any(self):
        """ Returns true if bool(e) is True for any element e in this Array. """
        return any(self)

    @property
    def max(self):
        return max(self)

    @property
    def argmax(self):
        """ Returns the index of the maximum value """
        return self.zipWithIndex.maxBy(lambda e: e[1])[0]

    @property
    def min(self):
        return min(self)

    @property
    def argmin(self):
        """ Returns the index of the minimum value """
        return self.zipWithIndex.minBy(lambda e: e[1])[0]

    @property
    def head(self):
        """ Selects the first element of this Array. """
        return self[0]

    @property
    def headOption(self, default=None):
        """
        Selects the first element of this Array if it has one,
        otherwise returns default.
        """
        return self.get(0, default)

    @property
    def last(self):
        """ Selects the last element of this Array. """
        return self[-1]

    @property
    def lastOption(self, default=None):
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
    def isEmpty(self):
        """ Returns whether this Array is empty. """
        return self.size == 0

    @property
    def isFinite(self):
        """
        Returns True if none of the elements are neither infinity nor NaN.
        """
        return self.forall(math.isfinite)

    @property
    def unique(self):
        """
        Selects unique values in this Array.
        Does not preserve order of elements.
        """
        return Array(set(self))

    @property
    def length(self):
        """ Number of elements in this Array. """
        return self.__len__()

    @property
    def size(self):
        """ Number of elements in this Array. """
        return self.__len__()

    @property
    def toInt(self):
        """
        Converts elements in this Array to integers.
        """
        try:
            return self.map(lambda e: ord(e) if isinstance(e, str) else int(e))
        except TypeError as err:
            raise TypeError("Expected an Array of numbers or characters") from err

    @property
    def toBool(self):
        """
        Converts elements in this Array to booleans.
        """
        return self.map(bool)

    @property
    def toArray(self):
        """ Converts all iterables in the Array to Arrays """
        return self.map(lambda e: Array(e).toArray if isinstance(e, Iterable) else e)

    @property
    def toChar(self):
        """
        Converts an Array of integers to chars.
        """
        return self.map(chr)

    @property
    def toStr(self):
        """
        Creates a string representation of the Array.
        See self.mkString for more functionality.
        """
        return "".join(str(v) for v in self)

    @property
    def toTuple(self):
        """ Returns a copy of the Array as an tuple. """
        return tuple(self.copy())

    @property
    def toSet(self):
        """ Returns set of the Array elements. Order is not preserved. """
        return set(self)

    @property
    def toIter(self):
        """ Returns an iterator for the Array."""
        for e in self:
            yield e

    @property
    def toDict(self):
        """ Creates a dictionary from an Array of tuples """
        return dict(self)

    @property
    def squeeze(self):
        """
        Returns the Array with the same elements, but with
        outermost singleton dimension removed (if exists).
        """
        if isinstance(self.headOption, Array.__baseIterables) and self.length == 1:
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
        r = Array([e for s in self for e in (s if isinstance(s, Iterable) else [s])])
        if r.forany(lambda e: isinstance(e, Iterable)):
            return r.flatten
        return r

    @property
    def range(self):
        """ Returns the indices of the Array"""
        return Array(range(self.size))

    @property
    def zipWithIndex(self):
        """ Zips the Array with its indices """
        return Array(enumerate(self))

    @property
    def unzip(self):
        """
        'Unzips' nested Arrays by unpacking its elements into a zip
        """
        if not self.forall(lambda e: isinstance(e, Iterable)):
            raise TypeError("Array elements must support iteration")
        return Array(zip(*self))

    @staticmethod
    def zeros(n):
        """ Returns a zero-filled Array of given length. """
        try:
            return Array([0] * n)
        except Exception as e:
            raise TypeError(
                "{} object cannot be interpreted as an integer".format(type(n).__name__)
            ) from e

    @staticmethod
    def arange(*args):
        """
        Returns an Array of evenly spaced values within given interval
        :param start: number, optional
        :param end: number
        :param step: number, optional
        """
        _len = len(args)
        if _len == 0:
            raise TypeError("Expected at least one argument")
        if _len > 3:
            raise TypeError("Expected at most 3 arguments")
        if _len == 1:
            return Array(range(math.ceil(args[0])))
        start, end, step = args if _len == 3 else args + (1,)
        return Array([start + step * i for i in range(math.ceil((end - start) / step))])

    @staticmethod
    def linspace(start, stop, num=50, endpoint=True):
        """
        Returns evenly spaced Array over a specified interval.
        :param start: number
        :param stop: number
        :param num: int, optional
        :param endpoint: bool, optional
        """
        num = operator.index(num)
        step = (stop - start) / max(num - bool(endpoint), 1)
        return Array(start + step * i for i in range(num))

    def __convert(self, e):
        return Array(e) if isinstance(e, Array.__baseIterables) else e

    def __validate_seq(self, e):
        if len(e) != self.size:
            raise ValueError(
                "The lengths of the sequences must match, got {} and {}".format(
                    self.size, len(e)
                )
            )

    def __operate(self, f, e):
        if isinstance(e, Iterable):
            self.__validate_seq(e)
            return Array(map(f, self, e))
        return Array(map(f, self, repeat(e)))

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
            raise TypeError(f"Can not concatenate {type(b).__name__} to Array")

    def __gt__(self, e):
        return self.__operate(gt, e)

    def __ge__(self, e):
        return self.__operate(ge, e)

    def __lt__(self, e):
        return self.__operate(lt, e)

    def __le__(self, e):
        return self.__operate(le, e)

    def __eq__(self, e):
        return self.__operate(eq, e)

    def __ne__(self, e):
        return self.__operate(ne, e)

    def __and__(self, e):
        return self.__operate(and_, e)

    def __or__(self, e):
        return self.__operate(or_, e)

    def __neg__(self):
        return self.mul(-1)

    def __invert__(self):
        if all(map(lambda e: isinstance(e, bool), self)):
            return self.map(lambda e: not e)
        return self.map(inv)

    def __hash__(self):
        return hash(self.toTuple)

    def __bool__(self):
        return self.all

    def __setitem__(self, i, e):
        if isinstance(i, Iterable):
            e = e if isinstance(e, Iterable) else repeat(e)
            for _i, _e in zip(i, e):
                super().__setitem__(_i, _e)
            return
        if isinstance(e, Array.__baseIterables):
            super().__setitem__(i, Array(e))
            return
        if isinstance(i, slice) and not isinstance(e, Iterable):
            e = [e] * len(range(*i.indices(self.size)))
        super().__setitem__(i, e)

    def __getitem__(self, key):
        if isinstance(key, Number):
            return super().__getitem__(key)
        if isinstance(key, slice):
            return Array(super().__getitem__(key))
        try:
            if all(map(lambda k: isinstance(k, bool), key)):
                if not (len(key) == self.size):
                    raise IndexError(
                        "Expected boolean array of size {}, got {}".format(
                            self.size, len(key)
                        )
                    )
                key = [i for i, k in enumerate(key) if k]
            return Array(map(super().__getitem__, key))
        except TypeError:
            return super().__getitem__(key)
