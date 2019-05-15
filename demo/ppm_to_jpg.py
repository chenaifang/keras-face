from PIL import Image
import os


def main():
    ppm_path = "./data/colerferet/"
    jpg_path = "./data/images/"
    for dirpath in os.listdir(r"./data/colerferet/"):
        os.mkdir(os.path.join(jpg_path, dirpath))
        for filename in os.listdir(r"./data/colerferet/"+ dirpath):
            filepath = os.path.join("./data/colerferet/" + dirpath + "/" + filename)
            newfilename1=filename.split('.')
            newfilename=newfilename1[0]
            #print(filepath)#./colorferet/dvd1/data/images/00001/00001_930831_fa_a.ppm.bz2
            newfilepath = os.path.join(jpg_path, dirpath + '/', newfilename+'.jpg' )
            img = Image.open(filepath)
            img.save(newfilepath)
            #img.show()

if __name__ == '__main__':
    main()