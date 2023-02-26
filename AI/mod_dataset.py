import os

# assign directory
directory = ['test/labels','train/labels','valid/labels']
#directory = ['files']

for dire in directory:
    for filename in os.listdir(dire):
        file = os.path.join(dire, filename)

        #if os.path.isfile(file):
            #print(file)

        with open(file, "r") as f:

            lines = f.readlines()

        with open(file, "w") as f:

            for line in lines:

                if line.startswith("0") or line.startswith("4"):

                    words = line.split()
                    words[0] = "0"
                    new_line = " ".join(words)
                    #print(new_line)

                    f.write(new_line+'\n')
            i=+1
    print(i)

