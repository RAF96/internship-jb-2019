
# Solve

MAXVALUE = ...

$ money_{n, r, i} $ - Цена, по которой купил человек n, из региона r, товар i. $ money \in [0, MAXVALUE) $

$ buy_{n, i} $ - бинарная переменная. Купил ли человек n, товар i.

$ greater\_equal_{n, k} $ где $ k \in {0, 1000, 2000, ... 6000}$ - бинарная переменная. Клиент потратил больше или равно k денег

$ less\_equal_{n, k} $ где $ k \in {-1, 999, 1999, 2999, 3999, 4999, 5999}$ - бинарная переменная. Клиент потратил <= k денег

$ nat_{n, r} $ - бинарная переменная, относится ли клиент n, к региону r

$ n \in [0, 95)$

$ r \in [0, 3) $

$ i \in [0, 5) $

[comment]: # ( $ p \in [0, MAX\_PRICE + 1) $)


|n| * |r| * |i| = 1425

# Functions

Потраченные деньги

$ \sum_{n,r,i} money_{n, r, i} $

Потраченные деньги на каждый товар в каждом регионе

$\forall n, r \sum_{i} money_{n, r, i} $

# Relationships

Связь money и greater\_equal

$ \forall n, k \sum_{r, i} money_{n, r, i} >= greater\_equal_{n, k} * k $ 

Связь money и less\_equal

$ \forall n, k \sum_{r, i} money_{n, r, i} <= MAXVALUE  - (MAXVALUE - k) * less\_equal_{n, k} $ 

Таблица 1

$ table1\_n= [32, 38, 10, 8, 2, 2, 3]$

$ \forall t : table1\_n\_sum= \sum_{i = 0}^{ i < t} table1\_n_i$

Связь из таблицы 1

$ \forall_{index} \sum_{n} greater\_equal_{n, k_{index}} = 95 - table1\_n\_sum_{index}$ 

$ \forall_{index} \sum_{n} less\_equal_{n, k_{index}} = table1\_n\_sum_{index} $ 

$ \forall n, k : less\_equal_{n, k}  + greater\_equal_{n, k} >= 1 $

$ \forall n : \sum_{k} less\_equal_{n, k}  + greater\_equal_{n, k} = len_k $

$ \forall n, k : less\_equal_{n, k}  <= less\_equal_{n, nextk} $

$ \forall n, k : greater\_equal_{n, k}  >= greater\_equal_{n, nextk} $

Таблицы 2

$ seg_1 = [33.000, 36.000] $

$ seg_2 = [120.000, 125.000] $

$ seg_3 = [10.000, 13.000] $

$ table2\_n = [22, 60, 13] $

Связь из таблицы 2

$ \forall r  : \sum_{n, i} money_{n, r, i} \in seg_r $


Связь переменных money с регионами

$ \forall n, r \sum_{i} money_{n, r, i} <=  nat_{n, r} * MAXVALUE $

Ограничения nat

$ \forall n \sum_{r} nat_{n, r} <= 1 $

$ \forall r \sum_{n} nat_{n, r} = table2\_n_{r} $

Таблица 3




$ seg_1 = [4.000, 6.000] $

$ seg_2 = [28.000, 30.000] $

$ seg_3 = [105.000, 109.000] $

$ seg_4 =  [16.000, 18.000] $

$ seg_5 =  [10.000, 12.000] $

$ table3\_n = [77, 87, 95, 91, 51] $


Ограничения money по сумме в соответствии с товаром

$ \forall i \sum_{n, r} money_{n, r, i} \in seg_{r} $

Ограничения buy

$ \forall i \sum_{n} buy_{n, i} = table3\_n_{i} $

Связь money с buy

$ \forall n, i : \sum_{r} money_{n, r, i} <=  buy_{n, i} * MAXVALUE $

Итоговое количество переменных: ~ 3800

Итоговое количество уравнений: ~ _

# Result

Проект не увенчался успехом, так как алгоритм не останавливается за разумное время (час). Скорее всего, из-за большого количества логических переменных библиотека не может дать результат.

# Code


```python
import time

```


```python
start = time.time()
start
```




    1558442393.567904




```python
from pulp import *
import numpy as np
import random
```


```python
MAXVALUE = 1e4
```


```python
len_n = 95
len_r = 3
len_i = 5
len_k = 7
greater_equal_border = list(map(lambda x: x * 1000, range(0, len_k)))
less_equal_border = list(map(lambda x: x * 1000 - 1, range(0, len_k)))
```


```python
greater_equal_border

```




    [0, 1000, 2000, 3000, 4000, 5000, 6000]




```python
less_equal_border
```




    [-1, 999, 1999, 2999, 3999, 4999, 5999]



# Var


```python
def money(*, n, r, i):
    return _money[n][r][i]
```


```python
_money = np.ndarray((len_n, len_r, len_i), dtype=object)

for n in range(len_n):
    for r in range(len_r):
        for i in range(len_i):
            _money[n, r, i] = LpVariable("money_{}_{}_{}".format(n, r, i), 0, MAXVALUE, cat='Continuous')
```


```python
def buy(*, n, i):
    return _buy[n][i]
```


```python
_buy = np.ndarray((len_n, len_i), dtype=object)

for n in range(len_n):
    for i in range(len_i):
        _buy[n, i] = LpVariable("buy_{}_{}".format(n, i), cat='Binary')
```


```python
def greater_equal(*, n, k):
    return _greater_equal[n][k]
```


```python
_greater_equal = np.ndarray((len_n, len_k), dtype=object)

for n in range(len_n):
    for k in range(len_k):
        _greater_equal[n, k] = LpVariable("greater_equal_{}_{}".format(n, k),cat='Binary')
```


```python
def less_equal(*, n, k):
    return _less_equal[n][k]
```


```python
_less_equal = np.ndarray((len_n, len_k), dtype=object)

for n in range(len_n):
    for k in range(len_k):
        _less_equal[n, k] = LpVariable("less_equal_{}_{}".format(n, k),cat='Binary')
```


```python
def nat(*, n, r):
    return _nat[n][r]
```


```python
_nat = np.ndarray((len_n, len_r), dtype=object)

for n in range(len_n):
    for r in range(len_r):
        _nat[n, r] = LpVariable("nat_{}_{}".format(n, r),cat='Binary')
```

# Relationships


```python
prob = LpProblem('task', LpMaximize)

prob += lpSum(_money)
```


```python
for n in range(len_n):
    for k in range(len_k):
        prob += lpSum(list(money(n=n, r=r, i=i) \
                           for r in range(len_r) \
                           for i in range(len_i))) \
                >= greater_equal(n=n, k=k) * greater_equal_border[k]
```


```python
for n in range(len_n):
    for k in range(len_k):
        prob += lpSum(list(money(n=n, r=r, i=i) \
                           for r in range(len_r) \
                           for i in range(len_i))) \
                <= MAXVALUE - \
                (MAXVALUE -less_equal_border[k]) \
                * less_equal(n=n, k=k)
```


```python
table1_n = [32, 38, 10, 8, 2, 2, 3]
```


```python
table1_n_sum = [sum(table1_n[0:i]) for i in range(len(table1_n) + 1)]
table1_n_sum
```




    [0, 32, 70, 80, 88, 90, 92, 95]




```python
for k in range(len_k):
    prob += lpSum([greater_equal(n=n, k=k) \
                   for n in range(len_n)]) \
            == 95 - table1_n_sum[k]
```


```python
for k in range(len_k):
    prob += lpSum([less_equal(n=n, k=k) \
                   for n in range(len_n)]) \
            == table1_n_sum[k]
```


```python
for n in range(len_n):
    prob += lpSum([less_equal(n=n, k=k) + greater_equal(n=n, k=k) \
                   for k in range(len_k)]) \
            == len_k
```


```python
for n in range(len_n):
    for k in range(len_k - 1):
        prob += less_equal(n=n, k=k) <= less_equal(n=n, k=k+1) 
```


```python
for n in range(len_n):
    for k in range(len_k - 1):
        prob += greater_equal(n=n, k=k) >= greater_equal(n=n, k=k+1) 
```


```python
prob.solve()
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-28-613465fcbb4d> in <module>
    ----> 1 prob.solve()
    

    ~/.conda/envs/common/lib/python3.7/site-packages/pulp/pulp.py in solve(self, solver, **kwargs)
       1662         #time it
       1663         self.solutionTime = -clock()
    -> 1664         status = solver.actualSolve(self, **kwargs)
       1665         self.solutionTime += clock()
       1666         self.restoreObjective(wasNone, dummyVar)


    ~/.conda/envs/common/lib/python3.7/site-packages/pulp/solvers.py in actualSolve(self, lp, **kwargs)
       1360     def actualSolve(self, lp, **kwargs):
       1361         """Solve a well formulated lp problem"""
    -> 1362         return self.solve_CBC(lp, **kwargs)
       1363 
       1364     def available(self):


    ~/.conda/envs/common/lib/python3.7/site-packages/pulp/solvers.py in solve_CBC(self, lp, use_mps)
       1419         cbc = subprocess.Popen((self.path + cmds).split(), stdout = pipe,
       1420                              stderr = pipe)
    -> 1421         if cbc.wait() != 0:
       1422             raise PulpSolverError("Pulp: Error while trying to execute " +  \
       1423                                     self.path)


    ~/.conda/envs/common/lib/python3.7/subprocess.py in wait(self, timeout)
        988             endtime = _time() + timeout
        989         try:
    --> 990             return self._wait(timeout=timeout)
        991         except KeyboardInterrupt:
        992             # https://bugs.python.org/issue25942


    ~/.conda/envs/common/lib/python3.7/subprocess.py in _wait(self, timeout)
       1622                         if self.returncode is not None:
       1623                             break  # Another thread waited.
    -> 1624                         (pid, sts) = self._try_wait(0)
       1625                         # Check the pid and loop as waitpid has been known to
       1626                         # return 0 even without WNOHANG in odd situations.


    ~/.conda/envs/common/lib/python3.7/subprocess.py in _try_wait(self, wait_flags)
       1580             """All callers to this function MUST hold self._waitpid_lock."""
       1581             try:
    -> 1582                 (pid, sts) = os.waitpid(self.pid, wait_flags)
       1583             except ChildProcessError:
       1584                 # This happens if SIGCLD is set to be ignored or waiting


    KeyboardInterrupt: 



```python
print( "Status:", LpStatus[prob.status])
```


```python
for v in prob.variables():
    print(v.name, "=", v.varValue)
```


```python
end = time.time()
end - start
```


```python

```


```python

```
