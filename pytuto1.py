# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 12:20:53 2017

@author: pegasus
"""

from __future__ import print_function
import torch

x=torch.rand(5,3)
print(x)
y=torch.rand(5,3)
print(y)
#print(x+y)
print(torch.add(x,y))