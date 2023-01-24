import json
import csv
 
with open('/home/jj/Qoakay_LTD/Semantic_Search/Official_Pilot_Data/5400_samples_data.json') as json_file:
    jsondata = json.load(json_file)
 
data_file = open('/home/jj/Qoakay_LTD/Semantic_Search/Official_Pilot_Data/5400_samples_data.csv', 'w', newline='')
csv_writer = csv.writer(data_file)
 
count = 0
for data in jsondata:
    if count == 0:
        header = data.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(data.values())
 
data_file.close()
