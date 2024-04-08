from matplotlib import pyplot as plt
from pandas import DataFrame
import pandas as pd

shared_memory_select = []
reg_select = []
point_score = {}
filter_scores = {}
with open('./ops/dense/point_score_43.out') as f:
    find_shared = False
    find_reg = False
    find_point = False
    for line in f.readlines():
        if line.find("/mnt/tvm/src/auto_scheduler/search_policy/sketch_policy.cc:969:") != -1:
            find_point = False
        if find_shared:
            shared_memory_select = line.strip().split(' ')
            find_shared = False
        if find_reg:
            reg_select = line.strip().split(' ')
            find_reg = False
        if find_point:
            id, score, filter_score, reg_filter_score = line.strip().split(' ')
            point_score[id] = float(score)
            filter_scores[id] = float(filter_score)
        if line.find("shared memory compute intensive select ids") != -1:
            find_shared = True
        if line.find("reg memory compute intensive select ids") != -1:
            find_reg = True
        if line.find("/mnt/tvm/src/auto_scheduler/search_policy/sketch_policy.cc:898:") != -1:
            find_point = True

plt.figure(figsize=(10, 10))
reg_ids = []
reg_score = []
for i in reg_select:
    reg_ids.append(filter_scores[i])
    reg_score.append(point_score[i])
plt.scatter(reg_ids, reg_score, c='r')
print(reg_score)
shared_ids = []
shared_score = []
for i in shared_memory_select:
    if i in reg_select:
        continue
    shared_ids.append(filter_scores[i])
    shared_score.append(point_score[i])
plt.scatter(shared_ids, shared_score, c='g')
point_ids = []
point_scores = []

for key in point_score:
    if key in shared_memory_select:
        continue
    point_ids.append(filter_scores[key])
    point_scores.append(point_score[key])
plt.scatter(point_ids, point_scores, c='b')
plt.savefig('point-score.png')

writer= pd.ExcelWriter("point-43.xlsx")
data = {'filtered points compute intensive': point_ids, 'filtered points gflops': point_scores}
df1 = DataFrame(data)
data = {'shared memory filter points compute intensive': shared_ids, 'shared memory filter points gflops': shared_score}
df2 = DataFrame(data)
data = {'reg filter points compute intensive': reg_ids, 'reg filter points gflops': reg_score}
df3 = DataFrame(data)
df1.to_excel(writer,sheet_name = "Sheet1",index= False)
df2.to_excel(writer,sheet_name = "Sheet2",index=False)
df3.to_excel(writer,sheet_name = "Sheet3",index=False)
writer.close()