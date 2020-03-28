---
title: "How slow is Python really?"
layout: post
categories: [general]
comments: true
branch: master
---

This is a very short post that compares plain `python` loops to `C` loops (sort of). We will write python code, compile it using the [Numba](https://numba.pydata.org) library and see the difference in both the implementations (interpreted vs compiled). <br>
I will not focus on writing elaborate explanations, but rather just the bits of code that we will need to show the difference.


```python
from numba import jit
import numpy as np
```

## How slow are plain python loops?

We'll use two very simplistic functions that can take an input to say how many times a loop should be run.


```python
def loop_n_times(n=100):
    x = 10
    for i in range(n):
        x += 5
```

The `numba` `@jit` decorator compiles the function it decorates into `C` code the first time it is run. So you'll see a noticeable lag when you run such a function for the first time, but every subsequent run is significantly faster as a result.


```python
@jit
def loop_n_times_fast(n=100):
    x = 10
    for i in range(n):
        x += 5
loop_n_times_fast(1) # Runnning once to compile the function
```

We will keep increasing the number of loops and see the difference in speed as we go on.


```python
%timeit -n 100 loop_n_times(100)
```

    6.61 µs ± 1.28 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
%timeit -n 100 loop_n_times_fast(100)
```

    366 ns ± 36.7 ns per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
print('Python is about',round(6.61 * 1000 / 366), 'times slower')
```

    Python is about 18 times slower



```python
%timeit -n 100 loop_n_times(10000)
```

    900 µs ± 90.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
%timeit -n 100 loop_n_times_fast(10000)
```

    189 ns ± 11.7 ns per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
print('Python is',round(900 * 1000 / 189), 'times slower')
```

    Python is 4762 times slower



```python
%timeit -n 100 loop_n_times(1000000)
```

    92.6 ms ± 7.22 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
%timeit -n 100 loop_n_times_fast(1000000)
```

    360 ns ± 30.7 ns per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
print('Python is',round(92.6 * 1000 * 1000 / 360), 'times slower')
```

    Python is 257222 times slower


Notice how the speed for executing python code keeps increasing as we increase the number of loops but the numba compiled code more or less runs at the same speed.<br>
In the end we see that for sufficient amount of processing, python can be a quarter million times slower than compiled `C` code. 

---
Anyway, I hope this short excerpt was informative for you.