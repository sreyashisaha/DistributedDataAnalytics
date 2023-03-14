#!/usr/bin/env python

import sys
import csv
file = csv.reader(sys.stdin,delimiter=',')
next(file)

for line in file:
    print('%s\t%s'%(line[0], line[3]))
    
