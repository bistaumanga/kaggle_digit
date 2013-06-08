import csv as csv 
import numpy as np

# reding from csv
csv_file_object = csv.reader(open('../data/train.csv', 'rb'))
header = csv_file_object.next()
data=[]
for row in csv_file_object:      
    data.append([map(int, row[1:]), [int(row[0])]])
print data[12]