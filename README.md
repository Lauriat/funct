# funct.Array

Array is a functional mutable sequence inheriting from Python's built-in list.
Array provides 100+ higher-order methods and more functionality to the built-in
list, making operations on sequences simpler and one-liners neater
with no third party packages required.

Array provides a combination of python built-ins, features found in NumPy arrays,
and higher-order methods common to functional languages
without the weird semantics of the builtins, still preserving
the same functionality and the dynamic nature of the built-in list.

[Documentation](https://Lauriat.github.io/funct/Array.html)

funct.Array is available on PyPi and can be installed with pip
```
$ pip install funct
```

Array Creation
-------
Arrays can be created either with multiple arguments or by providing a sequence
as an argument.

```python
>>> from funct import Array
>>> Array(1, 2, 3)
Array(1, 2, 3)
>>> Array([1, 2, 3])
Array(1, 2, 3)
```

An Array can also be initialized with the static `zeros` method or the `pad` method.

Python built-in sequences (including nested ones) lists, tuples and ranges are converted to
Arrays on instantiation. However, other iterables e.g. generators and numpy ndarrays
are converted to Arrays only if the argument consists of a single iterable. The elements
can be converted to Arrays by calling the `toArray` method.
```python
>>> Array(np.zeros(3))
Array(0.0, 0.0, 0.0)
>>> Array(np.zeros(3), np.zeros(3))
Array(array([0., 0., 0.]), array([0., 0., 0.])
>>> Array(np.zeros(3), np.zeros(3)).toArray
Array(Array(0.0, 0.0, 0.0), Array(0.0, 0.0, 0.0))
```

Arrays provide static methods `arange`, `linspace` and `logspace` for
creating linearly or logarithmically spaced Arrays.

Examples
-------

Chaining multiple functions with Arrays result in cleaner code without multiple
nested functions, e.g.
```python
a.zip(b).map(func1).filter(func2).forall(func3)

# vs. in traditional python

all(map(func3, filter(func2, map(func1, zip(a, b)))))
```
where `a` & `b` are Arrays and `func1`, `func2` & `func3` some functions.

##### Multiplying elements in a sequence with a constant

```python
#  In traditional python the multiplication could be implemented using list comprehensions as follows
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
>>> [a * b for a, b in zip(nums, nums2)]
[11, 24, 39, 56, 75]

#  With Arrays
>>> nums.mul(nums2)
Array(11, 24, 39, 56, 75)
```
Same syntax applies for all mathematical operators; `add`, `pow`, `mod`, `gt`, `lt`, etc.

##### Selecting values greater than some number
```python
#  Traditional python
>>> n = 2
>>> nums1 = [1, 2, 3, 4, 5]
>>> [x for x in nums if x > n]
[3, 4, 5]

#  With Arrays
>>> nums[nums > n]
Array(3, 4, 5)
```
##### Finding idex-wise maximum of sequences
```python
>>> nums1 = Array(1, 2, 3, 4, 5)
>>> nums2 = Array(5, 4, 3, 2, 1)
>>> nums1.zip(nums2).map(max)
Array(5, 4, 3, 4, 5)
```
##### Splitting an Array based on type
```python
>>> arr = Array(1, 2, "a", "b")
>>> arr.groupBy(type)[:, 1]  # group by type and select the 2nd element of the tuples
Array(Array(1, 2), Array('a', 'b'))
```

##### Multithreading/processing

Arrays also support parallel and concurrent execution.
Functions applied to Arrays can be parallelized with the `parmap` and
`parstarmap` methods. The same methods can be run asynchronously with the `asyncmap` and
`asyncstarmap` methods.
```python
>>> Array(1, 2, 3).parmap(some_heavy_func)
>>> Array(1, 2, 3).asyncmap(some_other_func)
```


Indexing
-------
Array indexing is a combination of standard Python sequence indexing and numpy-style
indexing.
Array supports
  - Standard Python indexing (single element indexing, slicing)
  - Index arrays
  - Boolean masking
  - Multidimensional indexing

### Examples

##### Standard Indexing
```python
>>> a = Array(1, 2, 3)
>>> a[0]
1
>>> a[:2]
Array(1, 2)
```

##### Index Arrays
```python
>>> a = Array('a', 'b', 'c', 'd')
>>> a[[1, 3]]
Array('b', 'd')
```

##### Boolean masking
```python
>>> a = Array(1, 2, 3, 4)
>>> a[[True, False, False, True]]
Array(1, 4)
```

##### Multidimensional indexing
```python
>>> a = Array((1, 2), (3, 4), (5, 6))
>>> a[:, 0]
Array(1, 3, 5)
```
Note that when indexing 'ragged' nested Arrays multidimensional indexing may
raise an `IndexError`, since Array does not care whether all the nested Arrays are
the same size, as opposed to numpy ndarrays.


#### Full documentation available [here](https://Lauriat.github.io/funct/Array.html).

Notes
-------
- Mathematical operations such as addition or multiplication can be done with the
  `add` and `mul` methods, not with the `+` and `*` operators to avoid confusion and to
  retain the behaviour of the built-in list.
- Inplace operations are postfixed with an underscore (e.g. `arr.abs_`). However,
  methods for adding elements to Arrays (`append`, `extend`, `insert`, etc.) are inplace
  by default. (**Note:** To be changed. In the next release the operations are inplace
  if `inplace=True` is passed to the methods.)
- Inplace operators are generally faster than out of place operations.
- Even though Array preserves nearly the same functionality
  as the built-in list, there are a few differences in their behaviour, the most
  important of which are
    - `==` (`__eq__`) Returns element-wise comparison.
    - `bool` (`__bool__`) Returns whether all elements evaluate to True.
    - Arrays are hashable. Note that this is implemented by using the Array's tuple representation in `__hash__`.
