from tvm.hardware import HardwareAPI, K80

a = K80()
b = HardwareAPI(a)
print(b.compute_capability)
