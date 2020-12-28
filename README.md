# Funct

A functional mutable sequence inheriting from Python's built-in list.
Array provides 100+ higher-order methods and more functionality to the built-in
list, making operations on sequences simpler and one-liners neater, with
no third party packages required.

Array provides a combination of python built-ins, features found in NumPy arrays,
and higher-order methods common to functional languages, still preserving
the same functionality and the dynamic nature of the built-in list.

[Documentation](https://Lauriat.github.io/funct/Array.html)

Funct.Array is available on PyPi and can be installed with pip
```python
$ pip install funct
```


Examples
-------

Chaining multiple functions with Arrays result in cleaner code without multiple
nested functions.
, e.g.
```python
a.zip(b).map(func1).filter(func2).forall(func3)

# vs. in traditional python

all(map(func3, filter(func2, map(func1, zip(a, b)))))
```

##### Multiplying elements in a sequence with a constant

```python
#  In traditional python we could implement it using list comprehensions as follows
>>> nums = [1, 2, 3, 4, 5]
>>> [a * 10 for a in nums]
[10, 20, 30, 40, 50]

#  With Arrays multiplication simplifies to
>>> from funct import Array
>>> nums = Array(nums)
>>> nums.mul(10)
Array(10, 20, 30, 40, 50)
```
##### Multiplying two sequences element-wise
```python
#  Traditional python
>>> nums2 = [11, 12, 13, 14, 15]
>>> [a * b for a, b in zip(nums, nums2]
[11, 24, 39, 56, 75]

#  With Arrays
>>> nums.mul(nums2)
Array(11, 24, 39, 56, 75)
```
Same syntax applies for all mathematical operators; `add`, `pow`, `mod`, `gt`, `lt`, etc.

##### Selecting even numbers
```python
#  Traditional python
>>> nums1 = [1, 2, 3, 4, 5]
>>> [x for x in nums if x % 2 == 0]
[2, 4]

#  With Arrays
>>> nums[nums.mod(2) == 0]
Array(2, 4)
```
##### Finding idex-wise maximum of sequences of uneven length
```python
>>> nums1 = Array(1, 2, 3, 4, 5)
>>> nums2 = Array(6, 5, 4, 3, 2, 1)
>>> nums1.zipAll(nums2, default=-float("inf").map(max)
Array(6, 5, 4, 4, 5, 1)
```
##### Splitting an Array based on type
```python
>>> arr = Array(1, 2, "a", "b")
>>> arr.groupBy(type).getItem(1)  # group by type and select the 2nd element of the tuples
Array(Array(1, 2), Array('a', 'b'))
```


Arrays provide static methods `arange`, `linspace` and `logspace` for
creating linearly or logarithmically spaced Arrays.
```python
>>> Array.linspace(0, 10, 5)
Array(0.0, 2.5, 5.0, 7.5, 10.0)
>>> Array.logspace(0, 4, 3)
Array(1.0, 100.0, 10000.0)
```

Arrays also support parallel computing.
Functions applied to Arrays can be parallelized with the `parmap` and
`parstarmap` methods.
```python
>>> Array(1,2,3).parmap(some_heavy_func)
```

#### Full documentation available [here](https://Lauriat.github.io/funct/Array.html).

Array Creation
-------
Arrays can be created either with multiple arguments or by providing a sequence
as an argument.

```python
>>> Array(1, 2, 3)
Array(1, 2, 3)
>>> Array([1, 2, 3])
Array(1, 2, 3)
```

An Array can also be initialized with the static `zeros` method or the `pad` method
```python
>>> Array.zeros(5).fill(5)
Array(5, 5, 5, 5, 5)
>>> Array().pad(5, value=5)
Array(5, 5, 5, 5, 5)
```

Python built-in sequences (including nested ones); lists, tuples and ranges are converted to
Arrays on instantiation. However, other iterables e.g. generators and numpy ndarrays
are converted to Arrays only if the argument consists of a single iterable. The elements
can be converted to Arrays by calling the `toArray` method.
```python
>>> Array(np.zeros(3))
Array(0.0, 0.0, 0.0)
>>> Array(np.zeros(3), np.zeros(3))
Array(array([0., 0., 0.]), array([0., 0., 0.])
>>> Array(np.zeros(3), np.zeros(3)).toArray
Array(Array(0.0, 0.0, 0.0], Array(0.0, 0.0, 0.0))
```


Notes
-------
- Mathematical operations such as addition or multiplication can be done with the
  `add` and `mul` methods, not with the `+` and `*` operators to avoid confusion and to
  retain the behaviour of the built-in list.
- Inplace operations in an Array are postfixed with a underscore (e.g. `arr.add_(x)`).
- Inplace operators are slower than out of place operations.
- Even though Array preserves nearly the same functionality
  as the built-in list, there are a few differences in their behaviour, the most
  important of which are
    - `==` (`__eq__`) Returns element-wise comparison.
    - `bool` (`__bool__`) Returns whether all elements evaluate to True.
    - Arrays are hashable. Note that this is implemented by using the Array's tuple representation in `__hash__`.



