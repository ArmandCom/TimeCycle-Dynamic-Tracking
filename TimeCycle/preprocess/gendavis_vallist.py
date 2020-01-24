import os
import numpy as np


outlist = '/data/Armand/TimeCycle/davis/DAVIS/vallist.txt'
imgfolder = '/data/Armand/TimeCycle/davis/DAVIS/JPEGImages/480p/'
lblfolder = '/data/Armand/TimeCycle/davis/DAVIS/Annotations/480p/'

jpglist = []

f1 = open('/data/Armand/TimeCycle/davis/DAVIS/ImageSets/2017/val.txt', 'r')
for line in f1:
    line = line[:-1]
    jpglist.append(line)
f1.close()


f = open(outlist, 'w')

for i in range(len(jpglist)):

    fname = jpglist[i]
    fnameim = imgfolder + fname + '/'
    fnamelbl= lblfolder + fname + '/'

    print(len(os.listdir(fnameim)) )

    if len(os.listdir(fnameim)) > 20:

        f.write(fnameim + ' ' + fnamelbl + '\n')


f.close()
