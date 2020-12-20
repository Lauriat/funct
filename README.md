# Funct

A functional mutable sequence inheriting from Python's built-in list.
Array provides higher-order methods and more functionality to the built-in
list, making operations on sequences simpler and one-liners neater.
No third party packages required.

Array provides a combination of python built-ins, functionality found in NumPy arrays,
and higher-order methods usually used in functional languages, however still retainging
the same functionality and the dynamic nature of the built-in list.

[Documentation](https://www.google.com)

Examples
-------

Multiplying elements in a sequence with a constant

```python
#  In traditional python we could implement it using list comprehensions as follows
>>> nums = [1, 2, 3, 4, 5]
>>> [a * 10 for a in nums]
[10, 20, 30, 40, 50]

#  With Arrays multiplication simplifies to
>>> nums = Array(nums)
>>> nums.mul(10)
Array(10, 20, 30, 40, 50)
```
Multiplying two sequences element-wise
```python
#  Traditional python
>>> nums2 = [11, 12, 13, 14, 15]
>>> [a * b for a, b in zip(nums, nums2]
[11, 24, 39, 56, 75]

#  With Arrays
>>> nums.mul(nums2)
Array(11, 24, 39, 56, 75)
```
Same syntax applies for all mathematical operations; `add`, `pow`, `mod`, `gt`, `lt`, etc.

Selecting even numbers
```python
#  Traditional python
>>> nums1 = [1, 2, 3, 4, 5]
>>> [x for x in nums if x % 2 == 0]
[2, 4]

#  With Arrays
>>> nums[nums.mod(2) == 0]
Array(2, 4)
```
Finding idex-wise maximum of sequences of uneven length
```python
#  Traditional python
>>> nums1 = [1, 2, 3, 4, 5]
>>> nums2 = [6, 5, 4, 3, 2, 1]
>>> min_ = min(min(nums1), min(nums2))
>>> out = []
>>> for i in range(max(len(nums1), len(nums2))):
        x1 = nums1[i] if i < len(nums1) else min_
        x2 = nums2[i] if i < len(nums2) else min_
        out.append(max(x1, x2))
>>> out
[6, 5, 4, 4, 5, 1]

#  With Arrays
>>> nums1.zipAll(nums2, default=min_).map(max)
Array(6, 5, 4, 4, 5, 1)
```
Arrays support also parallel programming.
Functions applied to Arrays can be parallelized with the `parmap` and
`parstarmap` methods.
```python
>>> Array(1,2,3).parmap(heavy_func)
```

Chaining multiple functions with Arrays result in cleaner code without multiple
nested functions.
, e.g.
```python
a.zip(b).map(func1).filter(func2).forall(func3)

# vs. in traditional python

all(map(func3, filter(func2, map(func1, zip(a, b)))))
```


#### Full documentation available [here](https://www.google.com).

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

Nested python built-in sequences i.e. lists, tuples and ranges are also converted to
Arrays. However, other iterables e.g. generators and numpy ndarrays
are converted to Arrays only if the argument consists of a single sequence. The elements
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
  `add` and `mul` methods, not with the symbols `+` and `*` to avoid confusion and to
  retain the functionality of the built-in list.
- Inplace operations in an Array are postfixed with a underscore (e.g. `arr.add\_(x)`).

- Even though Array preserves nearly the same functionality, 
  as the built-in list, there are a few differences in their behaviour, the most
  important of which are
    - `==` (`__eq__`) Returns element-wise comparison.
    - `bool` (`__bool__`) Returns whether all elements evaluate to True.
    - Arrays are hashable. Note that this is implemented by using the Array's tuple representation in `__hash__`.



