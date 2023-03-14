#!/usr/bin/env python
import sys
import csv

reader = csv.reader(sys.stdin, delimiter=',')
next(reader)
for line in reader:
    print(line[1], '\t', line[3])