import csv
inputs = ["100_samples_data_normalised.csv", "200_samples_data_normalised.csv", "300_samples_data_normalised.csv", "400_samples_data_normalised.csv", "500_samples_data_normalised.csv", "600_samples_data_normalised.csv", "700_samples_data_normalised.csv", "800_samples_data_normalised.csv", "900_samples_data_normalised.csv", "1000_samples_data_normalised.csv", "1100_samples_data_normalised.csv", "1200_samples_data_normalised.csv", "1300_samples_data_normalised.csv", "1400_samples_data_normalised.csv", "1500_samples_data_normalised.csv", "1600_samples_data_normalised.csv", "1700_samples_data_normalised.csv", "1800_samples_data_normalised.csv", "1900_samples_data_normalised.csv", "2000_samples_data_normalised.csv", "2100_samples_data_normalised.csv", "2200_samples_data_normalised.csv", "2300_samples_data_normalised.csv", "2400_samples_data_normalised.csv", "2500_samples_data_normalised.csv", "2600_samples_data_normalised.csv", "2700_samples_data_normalised.csv", "2800_samples_data_normalised.csv", "2900_samples_data_normalised.csv", "3000_samples_data_normalised.csv", "3100_samples_data_normalised.csv", "3200_samples_data_normalised.csv", "3300_samples_data_normalised.csv", "3400_samples_data_normalised.csv", "3500_samples_data_normalised.csv", "3600_samples_data_normalised.csv", "3700_samples_data_normalised.csv", "3800_samples_data_normalised.csv", "3900_samples_data_normalised.csv", "4000_samples_data_normalised.csv", "4100_samples_data_normalised.csv", "4200_samples_data_normalised.csv", "4300_samples_data_normalised.csv", "4400_samples_data_normalised.csv", "4500_samples_data_normalised.csv", "4600_samples_data_normalised.csv", "4700_samples_data_normalised.csv", "4800_samples_data_normalised.csv", "4900_samples_data_normalised.csv", "5000_samples_data_normalised.csv", "5100_samples_data_normalised.csv", "5200_samples_data_normalised.csv", "5300_samples_data_normalised.csv", "5400_samples_data_normalised.csv", "5500_samples_data_normalised.csv", "5600_samples_data_normalised.csv", "5700_samples_data_normalised.csv", "5800_samples_data_normalised.csv", "5900_samples_data_normalised.csv", "6000_samples_data_normalised.csv"]  # etc

# First determine the field names from the top line of each input file
# Comment 1 below
fieldnames = []
for filename in inputs:
  with open(filename, "r", newline="") as f_in:
    reader = csv.reader(f_in)
    headers = next(reader)
    for h in headers:
      if h not in fieldnames:
        fieldnames.append(h)

# Then copy the data
with open("out_v1.csv", "w", newline="") as f_out:   # Comment 2 below
  writer = csv.DictWriter(f_out, fieldnames=fieldnames)
  writer.writeheader()
  for filename in inputs:
    with open(filename, "r", newline="") as f_in:
      reader = csv.DictReader(f_in)  # Uses the field names in this file
      for line in reader:
        # Comment 3 below
        writer.writerow(line)
