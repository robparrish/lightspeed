#!/usr/bin/env python
import pyplugin as ls2
import lightspeed as ls

T = ls.Tensor.zeros((4,4))
T.identity()
ls2.print_tensor(T)
