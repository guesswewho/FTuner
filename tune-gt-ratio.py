from matplotlib import pyplot as plt
import numpy as np

shared_memory_select = []
reg_select = []
point_score = {}
filter_scores = {}
with open('./wlq.out') as f:
    find_start = False
    dietcode_result = []
    vendor_result = []
    for line in f.readlines():
        if line.find("start tune gt_ratio: ") != -1:
            find_start = True
            if len(dietcode_result) > 0:
                print(np.sum(np.array(dietcode_result)/np.array(vendor_result)))
                dietcode_result = []
                vendor_result = []
        if line.find("[logger.py +38, INFO] Vendor : ") != -1:
            vendor_result.append(float(line.strip().split("[logger.py +38, INFO] Vendor : ")[1].split("+")[0]))
        if line.find("[logger.py +38, INFO] DietCode : ") != -1:
            dietcode_result.append(float(line.strip().split("[logger.py +38, INFO] DietCode : ")[1].split("+")[0]))
