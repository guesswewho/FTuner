from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np

with open('wlq_dietcode_v100_bert_base.out') as f:
    dietcode_compile_time = []
    dietcode_inference_time = []
    pytorch_inference_time = []
    start_inference = False
    for line in f.readlines():
        if line.find("[logger.py +59, INFO] Auto-Scheduling Time for") != -1:
            time_str = line.split(" : ")[1].strip().split("s")[0].strip()
            dietcode_compile_time.append(float(time_str))
        if line.find("[logger.py +38, INFO] PyTorch : ") != -1:
            pytorch_inference_time.append(float(line.split("[logger.py +38, INFO] PyTorch : ")[1].strip().split("+")[0].strip()))
            start_inference = True
        if line.find("[logger.py +38, INFO] DietCode : ") != -1 and start_inference:
            dietcode_inference_time.append(float(line.split("[logger.py +38, INFO] DietCode : ")[1].strip().split("+")[0].strip()))

with open('wlq_v100_bert_base.out') as f:
    MIP_compile_time = []
    MIP_inference_time = []
    start_inference = False
    for line in f.readlines():
        if line.find("[logger.py +59, INFO] Auto-Scheduling Time for") != -1:
            time_str = line.split(" : ")[1].strip().split("s")[0].strip()
            MIP_compile_time.append(float(time_str))
        if line.find("[logger.py +38, INFO] PyTorch : ") != -1:
            start_inference = True
        if line.find("[logger.py +38, INFO] DietCode : ") != -1 and start_inference:
            MIP_inference_time.append(float(line.split("[logger.py +38, INFO] DietCode : ")[1].strip().split("+")[0].strip()))
print(np.sum(dietcode_compile_time))
print(np.sum(MIP_compile_time))
data = {'pytorch': pytorch_inference_time, 'dietCode': dietcode_inference_time, 'MIP': MIP_inference_time}
df = DataFrame(data)
df.to_excel('end-to-end-bert-base.xlsx')