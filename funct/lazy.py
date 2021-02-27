import math
import operator
import funct.Array as A


class ASeq:
    __slots__ = []

    def to_Array(self):
        """ Converts all iterables in the Array to Arrays """
        return A.Array(
            map(
                lambda e: A.Array(e).to_Array() if isinstance(e, A.Iterable) else e,
                self,
            )
        )

    def sum_(self, start=0):
        """ Returns the sum of the elements. """
        return sum(self, start)

    def product_(self, start=1):
        """ Returns the product of the elements. """
        return A.reduce(lambda a, b: a * b, self, start)

    def forall_(self, func):
        """
        Returns whether the specified predicate
        holds for all elements of the iterable.
        """
        return all(map(func, self))

    def forany_(self, func):
        """
        Returns whether the specified predicate
        holds for any element of the iterable.
        """
        return any(map(func, self))

    def foreach_(self, func):
        """ Applies `func` to each element of the iterable. """
        for e in self:
            func(e)

    def reduce_(self, func, init=None):
        """ Reduces the elements of the iterable using the specified operator. """
        if init is not None:
            return A.reduce(func, self, init)
        return A.reduce(func, self)

    def maxby_(self, func, **kwargs):
        """ Finds the maximum value measured by a function. """
        return max(self, key=func, **kwargs)

    def minby_(self, func, **kwargs):
        """ Finds the minimum value measured by a function. """
        return min(self, key=func, **kwargs)

    def argmax_(self):
        """ Returns the index of the maximum value """
        return self.enumerate_.maxby_(lambda e: e[1])[0]

    def argmin_(self):
        """ Returns the index of the minimum value """
        return self.enumerate_.minby_(lambda e: e[1])[0]

    def all_(self):
        """ Returns true if bool(e) is True for all elements in the iterable. """
        return all(self)

    def any_(self):
        """ Returns true if bool(e) is True for any element e in the iterable. """
        return any(self)

    def max_(self, **kwargs):
        return max(self, **kwargs)

    def min_(self, **kwargs):
        return min(self, **kwargs)

    def add_(self, e):
        """
        Lazy element-wise addition with given scalar or sequence.
        """
        return self.__lazy_operate(operator.add, e)

    def sub_(self, e):
        """
        Lazy element-wise subtraction with given scalar or sequence.
        """
        return self.__lazy_operate(operator.sub, e)

    def mul_(self, e):
        """
        Lazy element-wise multiplication with given scalar or sequence.
        """
        return self.__lazy_operate(operator.mul, e)

    def div_(self, e):
        """
        Lazy element-wise division with given scalar or sequence.
        """
        return self.__lazy_operate(operator.truediv, e)

    def pow_(self, e):
        """
        Raises elements of the iterable to given power,
        or sequence of powers, element-wise.
        """
        return self.__lazy_operate(operator, e)

    def mod_(self, e):
        """
        Computes the remainder between elements in the iterable
        and given scalar or sequence, element-wise.
        """
        return self.__lazy_operate(operator.mod, e)

    def bitwise_and_(self, e):
        """
        Computes the bit-wise AND between elements in the iterable
        and given scalar or sequence, element-wise.
        """
        return self.__lazy_operate(operator.and_, e)

    def bitwise_or_(self, e):
        """
        Computes the bit-wise OR between elements in the iterable
        and given scalar or sequence, element-wise.
        """
        return self.__lazy_operate(operator.or_, e)

    def abs_(self):
        """ Element-wise absolute value. """
        return Amap(abs, self)

    def floor_(self):
        """ Floors the elements of the iterable. """
        return Amap(math.floor, self)

    def ceil_(self):
        """ Ceils the elements of the iterable. """
        return Amap(math.ceil, self)

    def round_(self, d=0):
        """ Rounds the elements to the given number of decimals. """
        return Amap(lambda e: round(e, d), self)

    def gt_(self, e):
        """ Computes x > y element-wise """
        return self.__lazy_operate(operator.gt, e)

    def ge_(self, e):
        """ Computes x >= y element-wise """
        return self.__lazy_operate(operator.ge, e)

    def lt_(self, e):
        """ Computes x < y element-wise """
        return self.__lazy_operate(operator.lt, e)

    def le_(self, e):
        """ Computes x <= y element-wise """
        return self.__lazy_operate(operator.le, e)

    def eq_(self, e):
        """ Computes x == y element-wise """
        return self.__lazy_operate(operator.eq, e)

    def ne_(self, e):
        """ Computes x != y element-wise """
        return self.__lazy_operate(operator.ne, e)

    def isfinite_(self):
        """
        Tests element-wise whether the elements are neither infinity nor NaN.
        """
        return Amap(math.isfinite, self)

    def astype_(self, t):
        """
        Converts the elements in the iterable to given type.
        """
        return Amap(t, self)

    def int_(self):
        """ Converts elements in the iterable to integers. """
        try:
            return Amap(lambda e: ord(e) if isinstance(e, str) else int(e), self)
        except TypeError:
            raise TypeError("Expected an Array of numbers or characters") from None

    def float_(self):
        """ Converts elements in the iterable to floats. """
        return Amap(float, self)

    def bool_(self):
        """ Converts elements in the iterable to booleans. """
        return Amap(bool, self)

    def char_(self):
        """ Converts elements in the iterable to chars. """
        return Amap(chr, self)

    def clip_(self, _min, _max):
        """
        Clip the values in the iterable between the interval (`_min`, `_max`).
        """
        return Amap(lambda e: max(min(e, _max), _min), self)

    def map_(self, func):
        """ Lazy map """
        return Amap(func, self)

    def starmap_(self, func):
        """ Lazy starmap """
        return Amap(lambda a: func(*a), self)

    def filter_(self, func):
        """ Lazy filter """
        return Afilter(func, self)

    def takewhile_(self, func):
        """ Takes the longest prefix of elements that satisfy the given predicate. """
        return Aiter(A.itertools.takewhile(func, self))

    def dropwhile_(self, func):
        """ Drops the longest prefix of elements that satisfy the given predicate. """
        return Aiter(A.itertools.dropwhile(func, self))

    def zip_(self, *args):
        """ Lazy zip """
        return Azip(self, *args)

    def unzip_(self):
        """
        'Unzips' nested iterators by unpacking its elements into a zip.

        >>> Array((1, "a"), (2, "b")).unzip()
        Array(Array(1, 2), Array('a', 'b'))
        """
        return Azip(*self)

    def zip_all_(self, *args, default=None):
        """
        Zips the iterables. If the iterables are
        of uneven length, missing values are filled with default value.
        """
        return Aiter(A.itertools.zip_longest(self, *args, fillvalue=default))

    def next_(self, *args):
        """
        Returns the first element of the iterator.
        If the iterator is exhausted returns default if given.
        """
        return next(self, *args)

    @property
    def enumerate_(self):
        """ Zips the iterator with its indices """
        return Aenum(self)

    def __lazy_operate(self, f, e):
        fn = (
            lambda x, y, f=f: A.Array(A.Array(x)._apply(f, y))
            if isinstance(x, A.Iterable)
            else f(x, y)
        )
        if isinstance(e, A.Iterable):
            return Amap(fn, self, e)
        return Amap(fn, self, A.itertools.repeat(e))

    def copy_(self, n=2):
        """
        Returns n independent iterators from the iterable.
        Exhausts the original iterator.
        """
        return tuple(Aiter(e) for e in A.itertools.tee(self, n))


class AFunc:
    __slots__ = []

    def result(self):
        return A.Array(self)


class Amap(ASeq, AFunc, map):
    __slots__ = []


class Afilter(ASeq, AFunc, filter):
    __slots__ = []


class Azip(ASeq, AFunc, zip):
    __slots__ = []


class Aenum(ASeq, AFunc, enumerate):
    __slots__ = []


class Aiter(ASeq, AFunc):
    __slots__ = "__val"

    def __init__(self, val):
        self.__val = val

    def __next__(self):
        return next(self.__val)

    def __iter__(self):
        return iter(self.__val)
