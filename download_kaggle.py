import kagglehub

path = kagglehub.dataset_download("debs2x/gamelytics-mobile-analytics-challenge")
print("Path to dataset files:", path)
import os

path = r"C:\Users\irmak\.cache\kagglehub\datasets\debs2x\gamelytics-mobile-analytics-challenge\versions\2"

files = os.listdir(path)

for f in files:
    print(f)

import os

path = r"C:\Users\irmak\.cache\kagglehub\datasets\debs2x\gamelytics-mobile-analytics-challenge\versions\2"

print(os.listdir(path))
