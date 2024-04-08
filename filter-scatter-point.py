from matplotlib import pyplot as plt
from pandas import DataFrame

shared_memory_select = []
reg_select = []
point_score = {}
point_filter_score = {}
flops_scores = []
filter_scores = []
with open('./ops/dense/point_score_81.out') as f:
    find_shared = False
    find_reg = False
    find_point = False
    for line in f.readlines():
        if line.find("/mnt/tvm/src/auto_scheduler/search_policy/sketch_policy.cc:961:") != -1:
            find_point = False
        if find_shared:
            shared_memory_select = line.strip().split(' ')
            find_shared = False
        if find_reg:
            reg_select = line.strip().split(' ')
            find_reg = False
        if find_point:
            id, score, filter_score = line.strip().split(' ')
            flops_scores.append(float(score))
            filter_scores.append(float(filter_score))
        if line.find("shared memory compute intensive select ids") != -1:
            find_shared = True
        if line.find("reg memory compute intensive select ids") != -1:
            find_reg = True
        if line.find("/mnt/tvm/src/auto_scheduler/search_policy/sketch_policy.cc:890:") != -1:
            find_point = True

plt.figure(figsize=(10, 10))
plt.scatter(filter_scores, flops_scores, c='b')
plt.savefig('point-score.png')
data = {'gflops': flops_scores, 'compute intensive': filter_scores}
df = DataFrame(data)
df.to_excel('point_81.xlsx')
