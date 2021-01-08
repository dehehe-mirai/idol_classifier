import cv2
import bulk_convert
import os
import numpy as np
import sys
import shutil
from glob import glob
import tkinter as tk
from tkinter import Tk, filedialog, messagebox, ttk
import label_image
import random

root = Tk()
root.title("Million live Idol Cl@ssifier")
root.geometry("700x100+100+100")
root.resizable(False, False)

def predict(filepath, dst="classified", move=False):

    origs = [y for x in os.walk(filepath) for y in glob(os.path.join(x[0], '*.*'))]
    tk.Label(root, text="파일 분류 중...").grid(sticky="W", row=0,column=0)

    progress = 0
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=len(origs))
    progress_bar.grid(sticky="W", row=1, column=0)
    file_text = tk.StringVar()
    file_label = tk.Label(root, textvariable=file_text).grid(sticky="W", row=3,column=0)

    root.pack_slaves()

    for file in origs:
        if os.path.splitext(file)[1] not in ["", ".png", ".jpg", ".jfif"]:
            continue
        file_text.set(file)
        newname = os.path.join(os.path.dirname(file),str(random.randint(1, 10000000)) + ".png")
        os.rename(file, newname)
        print(f"{file}>{newname}")
        file = newname
        try:
            print("reading: " + file)
            orig = cv2.imread(file)
            if len(orig) == 0:
                print("read error: " + file)
        except Exception as e:
            print(e)
            print("Error occured in load image!")

        try:
            images = bulk_convert.bulk_convert_image_ld(orig)
            if len(images) == 0:
                print("얼굴이 발견되지 않음!")
            for image in images:
                label = label_image.inference(image)
                # cv2.imshow(label, image)
                if move:
                    target_path = os.path.join(dst, label)
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    filename = os.path.basename(file)
                    shutil.copy(file, os.path.join(target_path, filename))


                # # display the predictions with the image
                # cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
                #             cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

                # cv2.imshow("Classification", orig)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        except Exception as e:
            print("Error occured!")
            print(e)
            progress += 1
            progress_var.set(progress)

            continue
        progress += 1
        progress_var.set(progress)

        root.update()

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

if __name__ == "__main__":
    currdir = os.getcwd()
    messagebox.showinfo("안내", "폴더 경로명에 한글이 있을 시 오류가 발생할 수 있습니다.")

    target_dir = filedialog.askdirectory(parent=root, initialdir=currdir, title='분류할 폴더를 선택하세요')
    if not target_dir:
        sys.exit(0)
    
    dst_dir = filedialog.askdirectory(parent=root, initialdir=currdir, title='분류된 파일이 이동될 폴더를 선택하세요')
    if not dst_dir:
        sys.exit(0)
    
    predict(target_dir, dst = dst_dir, move=True)