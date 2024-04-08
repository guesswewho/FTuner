from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np

compile_time = []
group_nums = []
for i in range(2, 74, 2):
    with open('wlq_v100_group_dense_'+str(i)+'.out') as f:
        for line in f.readlines():
            if line.find("[logger.py +59, INFO] Auto-Scheduling Time for") != -1:
                time_str = line.split(" : ")[1].strip().split("s")[0].strip()
                compile_time.append(float(time_str))
                break
        group_nums.append(i)
for i in range(74, 131, 3):
    with open('wlq_v100_group_dense_'+str(i)+'.out') as f:
        for line in f.readlines():
            if line.find("[logger.py +59, INFO] Auto-Scheduling Time for") != -1:
                time_str = line.split(" : ")[1].strip().split("s")[0].strip()
                compile_time.append(float(time_str))
                break
        group_nums.append(i)
data = {'groups num': group_nums, 'compile time': compile_time}
df = DataFrame(data)
df.to_excel('groups-compile-time.xlsx')