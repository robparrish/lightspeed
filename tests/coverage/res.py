#!/usr/bin/env python
import lightspeed as ls

res = ls.ResourceList.build(1024,1024)

print(res)

gpu = res.gpus[0]
print(gpu)

print(gpu.name)
print('%.1f' %  gpu.CC)
