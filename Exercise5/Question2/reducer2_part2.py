#!/usr/bin/env python
import sys
import csv
delay_list = []
current_airport = None
airport=[]
average =[]

for line in sys.stdin:
    # print("line is",line)
    line = line.strip()
    origin, delay = line.split('\t')
    if current_airport is not None:
        if current_airport == origin:
            delay_list.append(int(float(delay)))
        else:
            if delay_list:
                airport.append(current_airport)
                average.append((sum(delay_list)/len(delay_list)))
                # print(current_airport, max(delay_list), min(delay_list),(sum(delay_list)/len(delay_list)))
                current_airport=origin
                delay_list=[]
    else:
        current_airport=origin
        delay_list.append(int(float(delay)))

sort_function = zip(airport, average)
sorted_values = sorted(sort_function, key=lambda val: val[1], reverse=True)[:10]
# print(sorted_values)
print("Airport",'\t',"Avg Arrival Delay",'\n')
for key in sorted_values:
    print(*key, sep='\t\t')
    print('\n')
    