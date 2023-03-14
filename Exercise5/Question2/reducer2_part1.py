#!/usr/bin/env python  
import sys
import csv
delay_list = []
current_airport = None
print("Origin",'  ',"Maximum",'  ',"Minimum",'  ',"Average",'\n')
for line in sys.stdin:
    # print("line is",line)
    line = line.strip()
    origin, delay = line.split('\t')
    if current_airport is not None:
        if current_airport == origin:
            delay_list.append(int(float(delay)))
        else:
            if delay_list:
                print(current_airport,'      ', max(delay_list),'    ', min(delay_list),'    ',(sum(delay_list)/len(delay_list)))
            current_airport=origin
            delay_list=[]
    else:
        current_airport=origin
        delay_list.append(int(float(delay)))
    