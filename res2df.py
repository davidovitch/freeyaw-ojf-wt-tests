# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 17:40:17 2015

@author: dave
"""
import sys

import pandas as pd

import ojfresult

# cleanup path
blacklist = [ '/home/dave/Repositories/public/MMPE',
              '/home/dave/Repositories/DTU/prepost',
              '/home/dave/Repositories/DTU/pythontoolbox/fatigue_tools',
              '/home/dave/Repositories/DTU/pythontoolbox']
rm = []
for path_rm in blacklist:
    for i, path in enumerate(sys.path):
        if path == path_rm:
            print 'removed from path: %s' % path
            sys.path.pop(i)
            break


if __name__ == '__main__':
    dummy = None
