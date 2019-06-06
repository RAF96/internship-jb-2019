
# Solve

MAXVALUE = 1e6

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

$ \forall n \sum_{r} nat_{n, r} = 1 $

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

# Code


```python
import time

```


```python
from pulp import *
import numpy as np
import random
import pandas as pd
```


```python
MAXVALUE = 1e6
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
seg_2 = [[33e3, 36e3],
       [120e3, 125e3],
       [10e3, 13e3]]
table2_n = [22, 60, 13]
```


```python
for r in range(len_r):
    prob += lpSum([money(n=n, r=r, i=i) for n in range(len_n) for i in range(len_i)] ) >= seg_2[r][0]
    prob += lpSum([money(n=n, r=r, i=i) for n in range(len_n) for i in range(len_i)] ) <= seg_2[r][1]
```


```python
for r in range(len_r):
    for n in range(len_n):
        prob += lpSum([money(n=n, r=r, i=i) for i in range(len_i)]) <= nat(n=n, r=r) * MAXVALUE
```


```python
for n in range(len_n):
    prob += lpSum([nat(n=n, r=r) for r in range(len_r)]) == 1
```


```python
for r in range(len_r):
    prob += lpSum([nat(n=n, r=r) for n in range(len_n)]) == table2_n[r]
```


```python
seg_3 = [[4e3,6e3], 
         [28e3,30e3], 
         [105e3,109e3], 
         [16e3,18e3], 
         [10e3,12e3]] 
table3_n = [77,87,95,91,51]
```


```python
for i in range(len_i):
    prob += lpSum([money(n=n,r=r,i=i) for r in range(len_r) for n in range(len_n)]) >= seg_3[i][0]
    prob += lpSum([money(n=n,r=r,i=i) for r in range(len_r) for n in range(len_n)]) <= seg_3[i][1]
```


```python
for i in range(len_i):
    prob += lpSum([buy(n=n,i=i) for n in range(len_n)]) == table3_n[i]
```


```python
for i in range(len_i):
    for n in range(len_i):
        prob += lpSum([money(n=n,r=r,i=i) for r in range(len_r)]) <= buy(n=n, i=i) * MAXVALUE
```


```python
prob.solve()
```




    1




```python
print( "Status:", LpStatus[prob.status])
```

    Status: Optimal



```python
# for v in prob.variables():
#     print(v.name, "=", v.varValue)
```

# Check


```python
# First condition
check_table1 = [0] * 7
for n in range(len_n):
    
    def index_of_check_table(value):
        index_of_check_table = min(int(value / 1e3), 6)
        return index_of_check_table
    
    res = sum(money(n=n, r=r, i=i).value() for i in range(len_i) for r in range(len_r))
    check_table1[index_of_check_table(res)] += 1
check_table1
```




    [32, 38, 10, 8, 2, 2, 3]




```python
# Second condition
check_table2_sum = [0] * 3
check_table2_num = [0] * 3

for r in range(len_r):
    for n in range(len_n):
        check_table2_num[r] += nat(n=n, r=r).value()
        check_table2_sum[r] += sum(money(n=n, r=r, i=i).value() for i in range(len_i))
        num_of_nat = sum(nat(n=n, r=r).value() for r in range(len_r))
        if num_of_nat != 1:
            print("Warning: number of nation for {} person is {}".format(n, num_of_nat))
        
print(check_table2_num)
print(check_table2_sum)
```

    [22.0, 60.0, 13.0]
    [36000.0, 125000.0, 13000.0]



```python
# Third condition
check_table_3_sum = [0] * 5
check_table_3_num = [0] * 5
for i in range(len_i):
    for n in range(len_n):
        check_table_3_num[i] += buy(n=n, i=i).value()
        check_table_3_sum[i] += sum(money(n=n,r=r,i=i).value() for r in range(len_r))
        
print(check_table_3_num)
print(check_table_3_sum)
```

    [77.0, 87.0, 95.0, 91.0, 51.0]
    [6000.0, 30000.0, 108000.0, 18000.0, 12000.0]



```python
# Result
print(sum(money(n=n, r=r, i=i).value() for i in range(len_i) for r in range(len_r) for n in range(len_n)))
```

    174000.0



```python
data = pd.DataFrame([[sum(money(n=n, r=r, i=i).value()
                     for n in range(len_n)) for i in range(len_i)] for r in range(len_r)], 
                    index=['EU', 'US', 'ROW'],
                    columns=['A', 'B', 'C', 'D', 'X'])
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EU</th>
      <td>2001.0</td>
      <td>10007.0</td>
      <td>23992.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>US</th>
      <td>2999.0</td>
      <td>16993.0</td>
      <td>77007.0</td>
      <td>16001.0</td>
      <td>12000.0</td>
    </tr>
    <tr>
      <th>ROW</th>
      <td>1000.0</td>
      <td>3000.0</td>
      <td>7001.0</td>
      <td>1999.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



# Summary 

Результаты подходят под все ограничения.
Модель находит результаты от 163e3 до 174e3.

# Особенности

Визуально заметно, что переменная buy, то-есть ограничение по количесвту покупок, плохо взаимосвязано с количество потраченных денег. Так, встречаются данные, в которых человек купил все товары, но при этом суммарно потратил 0. То-есть как-будто получил все бесплатно. С другой стороны, по имеющимся данным, кажется такая ситуация вполне возможна и ей не стоит пренебрегать или ограничивать.


```python

```
