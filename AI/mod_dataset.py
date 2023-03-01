#!/usr/bin/env python3

"""
Modify Datasets (remove or combine class labels)
"""


import os

# assign directory
directory = ['Stanford_Car.v10-accurate-model_mergedallclasses-augmented_by3x.yolov8/test/labels',
             'Stanford_Car.v10-accurate-model_mergedallclasses-augmented_by3x.yolov8/train/labels',
             'Stanford_Car.v10-accurate-model_mergedallclasses-augmented_by3x.yolov8/valid/labels']
# directory = ['']

for dire in directory:
    for filename in os.listdir(dire):
        file = os.path.join(dire, filename)

        #if os.path.isfile(file):
            #print(file)

        with open(file, "r") as f:
            lines = f.readlines()

        with open(file, "w") as f:

            for line in lines:

                # if line.startswith("0") or line.startswith("4"):
                if line.startswith("1"):

                    # Replace class number with 0
                    words = line.split()
                    words[0] = "0"
                    new_line = " ".join(words)
                    #print(new_line)

                    f.write(new_line+'\n')
                i=+1
    print(i)

