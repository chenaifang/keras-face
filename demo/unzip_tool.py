import sys
import os
import bz2
from bz2 import decompress

def main():
    path = "./data/images/"
    un_path = "./data/dlib-align-image/"
    for dirpath in os.listdir(r"./data/images"):
        os.mkdir(os.path.join(un_path, dirpath))
        for filename in os.listdir(r"./data/images/"+dirpath):
            filepath = os.path.join("./data/images/"+dirpath+"/"+filename)
            #print(filepath)#./colorferet/dvd1/data/images/00001/00001_930831_fa_a.ppm.bz2
            newfilepath = os.path.join(un_path, dirpath +'/',filename + '.ppm')
            with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(data)

if __name__ == '__main__':
    main()