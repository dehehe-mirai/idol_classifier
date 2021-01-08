import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from PIL import Image

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

model_file = "model/model.tflite"
label_file = "label/labels.txt"
labels = load_labels(label_file)
num_threads = 4

interpreter = None

def inference(img):
    if len(labels) != 56:
        print(labels)
        print("labels length!")
    global interpreter
    # load model
    if interpreter == None:
        interpreter = tflite.Interpreter(
            model_path=model_file, num_threads=num_threads)
        interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = cv2.resize(img, (height, width))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    # add N dim
    
    # input_data = np.expand_dims(img, axis=0)
    input_data = np.array([img.astype(np.float32)])
    # input_mean = 0
    # input_std = 255.0

    # if floating_model:
    #     input_data = np.float32(input_data)#- input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_id = np.argmax(output_data, axis=-1)
    results = np.squeeze(output_data)
    
    top_k = results.argsort()[-5:][::-1]
    show_result = False

    if show_result:
        for i in top_k:
            if floating_model:
                print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            else:
                print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

        print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    try:
        return labels[top_k[0]]
    except:
        print("Error occured!")
        print(top_k)
        print(labels)
        print(len(labels))
        print(top_k[0])
        print(labels[top_k[0]])
        return None

# 성능평가
if __name__ == "__main__":
    import bulk_convert
    import os
    import matplotlib.pylab as plt

    bulk = True

    if bulk:
        path = "raw"
        dir_labels = os.listdir(path)
        for label in dir_labels:
            answer = os.path.basename(label)
            i = 0
            n = 0
            dir = os.listdir(os.path.join(path,label))
            if len(dir) == 0:
                continue
            print(f"{label}: {len(dir)}")
            plt.figure(figsize=(10,9))
            plt.subplots_adjust(hspace=0.5)
            for file in dir:
                img = cv2.imread(os.path.join(path, label, file))
                faces = bulk_convert.bulk_convert_image_ld(img)
                predicted = []
                
                for image in faces:
                    result = inference(image)
                    predicted.append(result)
                    # if n < 100:
                    #     plt.subplot(10,10,n+1)
                    #     plt.imshow(image)
                    #     plt.title(result)
                    #     plt.axis('off')
                    #     n += 1
                if answer in predicted:
                    i += 1
            # plt.show()
            print(f"{label}: n: {n}, i: {i}, rate: {i/len(dir)}")
    else:
        path = "raw-fhd/Sayoko/1335764.png"
        img = cv2.imread(path)
        faces = bulk_convert.bulk_convert_image_ld(img)
        predicted = []
        for image in faces:
            predicted.append(inference(image))
        print(predicted)
        
