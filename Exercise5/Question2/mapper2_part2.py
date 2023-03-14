#!/usr/bin/env python       

import sys
import csv

file = csv.reader(sys.stdin,delimiter=',')
# print(file)
next(file)
for line in file:
    FL_DATE, OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, DEST, DEP_TIME, DEP_DELAY, ARR_TIME, ARR_DELAY = line
    print('%s\t%s' % (ORIGIN, ARR_DELAY))