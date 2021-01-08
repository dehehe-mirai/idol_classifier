import cv2
import os
import sys
import random
import json
from core.detector import LFFDDetector
from glob import glob

CONFIG_PATH = "configs/anime.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
    detector = LFFDDetector(config, use_gpu=False)
def bulk_convert_image_ld(image):
    height, width, channels = image.shape
    faces = detector.detect(image)
    i=0
    cropped_faces = []
    for data in faces:
        x = data["xmin"]
        y = data["ymin"]
        xm = data["xmax"]
        ym = data["ymax"]
        crop_img = image[y:ym, x:xm]
        cropped_faces.append(crop_img)
    return cropped_faces



def bulk_convert_image(image, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    # Create classifier
    cascade = cv2.CascadeClassifier(cascade_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray,
                                    # detector options
                                    scaleFactor = 1.1,
                                    minNeighbors = 5,
                                    minSize = (30, 30))
    images = []
    for (x, y, w, h) in faces:
        crop_img = image[y:y+h, x:x+w]
        images.append(crop_img)
    return images

def bulk_convert_file(image_file, dst, cascade_file = "lbpcascade_animeface.xml"):
    target_path = "/".join(image_file.strip("/").split('/')[1:-1])
    target_path = os.path.join(dst, target_path) + "/"
    print("target:" + target_path)
    cascade = cv2.CascadeClassifier(cascade_file)

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    image = cv2.imread(image_file)
    return bulk_convert_image(image)

total = []
def bulk_convert(src, dst, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    # Create classifier
    cascade = cv2.CascadeClassifier(cascade_file)
    files = [y for x in os.walk(src) for y in glob(os.path.join(x[0], '*.*'))]
    for image_file in files:
        newname = os.path.join(src,str(random.randint(1, 10000000)) + ".png")
        os.rename(image_file, newname)
        image_file = newname
        try:
            target_path = "/".join(image_file.strip("/").split('/')[1:-1])
            target_path = os.path.join(dst, target_path) + "/"
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            image = cv2.imread(image_file)
            height, width, channels = image.shape
            if height > 1024:
                image = cv2.resize(image, (int(width * 1),int(height * 1)))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = cascade.detectMultiScale(gray,
                                            # detector options
                                            scaleFactor = 1.1,
                                            minNeighbors = 5,
                                            minSize = (30, 30))
            for (x, y, w, h) in faces:
                crop_img = image[y:y+h, x:x+w]
                filename = os.path.basename(image_file).split('.')[0]
                cv2.imwrite(
                    os.path.join(target_path, filename + ".png"),
                    crop_img
                )
        except Exception as e:
            print(e)
            print(image_file, " Process Failed!")
    if not os.path.exists(dst):
        return
    old_dir = os.listdir(src)
    new_dir = os.listdir(dst)
    success = len(new_dir)/len(old_dir)
    total.append(success)
    print(f"{src} Success Rate: {success}")

def bulk_convert_ldetector(src, dst):
    files = [y for x in os.walk(src) for y in glob(os.path.join(x[0], '*.*'))]
    for image_file in files:
        newname = os.path.join(src,str(random.randint(1, 10000000)) + ".png")
        os.rename(image_file, newname)
        image_file = newname
        try:
            target_path = "/".join(image_file.strip("/").split('/')[1:-1])
            target_path = os.path.join(dst, target_path) + "/"
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            image = cv2.imread(image_file)
            height, width, channels = image.shape
            if height > 1024:
                image = cv2.resize(image, (int(width * 1),int(height * 1)))

            faces = detector.detect(image)
            i=0
            for data in faces:
                x = data["xmin"]
                y = data["ymin"]
                xm = data["xmax"]
                ym = data["ymax"]
                crop_img = image[y:ym, x:xm]
                filename = os.path.basename(image_file).split('.')[0]
                cv2.imwrite(
                    os.path.join(target_path, filename + str(i) + ".png"),
                    crop_img
                )
                i = i + 1
        except Exception as e:
            print(e)
            print(image_file, " Process Failed!")
    if not os.path.exists(dst):
        return
    old_dir = os.listdir(src)
    new_dir = os.listdir(dst)
    success = len(new_dir)/len(old_dir)
    total.append(success)
    print(f"{src} Success Rate: {success}")

def main():
    # if len(sys.argv) != 3:
    #     sys.stderr.write("usage: bulk_convert.py <source-dir> <target-dir>\n")
    #     sys.exit(0)
    #bulk_convert(sys.argv[1], sys.argv[2])
    for dir in os.listdir("./raw-fhd"):
        bulk_convert_ldetector(os.path.join("raw-fhd",dir), os.path.join("cropped-fhd-ld",dir))
    if len(total) != 0:
        print(f"Average rate: {sum(total)/len(total)}")

if __name__ == '__main__':
    main()
