import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import PIL.Image as Image
import time

batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "cropped",
    validation_split=0.2,
    subset="training",
    seed=456,
    image_size=(img_height, img_width),
    batch_size=batch_size)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
print("=================================")
class_names = np.array(train_ds.class_names)
print(class_names)
import label_image
pre_label = label_image.load_labels("label/labels.txt")
for i in range(0,len(class_names)):
    assert class_names[i] == pre_label[i]
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
for image_batch, labels_batch in train_ds:
    print(image_batch)
    print(image_batch.shape)
    print(labels_batch.shape)
    break
New = False
model = None
if New:
    feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False)
    feature_batch = feature_extractor_layer(image_batch)
    print(feature_batch.shape)
    num_classes = len(class_names)

    model = tf.keras.Sequential([
        normalization_layer,
        feature_extractor_layer,
        tf.keras.layers.Dense(num_classes)
    ])


    predictions = model(image_batch)
    predictions.shape

    model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])
    model.summary()
else:
    export_path = "idol_models/{}".format(1609991292)
    model = tf.keras.models.load_model(export_path)

class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []
    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()

batch_stats_callback = CollectBatchStats()
if New:
    history = model.fit(train_ds, epochs=10,
                        callbacks=[batch_stats_callback])

    # Train Results
    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0,2])
    plt.plot(batch_stats_callback.batch_losses)

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0,1])
    plt.plot(batch_stats_callback.batch_acc)
    plt.show()

# Prediction
print("==============Prediction=============")
print(image_batch[0])
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]

# Prediction Results
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(predicted_label_batch[n].title())
    plt.axis('off')

_ = plt.suptitle("Model predictions")

plt.show()

t = time.time()

if New:
    export_path = "idol_models/{}".format(int(t))
    model.save(export_path)

export_path

reloaded = tf.keras.models.load_model(export_path)

# result_batch = model.predict(image_batch)
# reloaded_result_batch = reloaded.predict(image_batch)
# abs(reloaded_result_batch - result_batch).max()

converter = tf.lite.TFLiteConverter.from_saved_model(export_path) # path to the SavedModel directory
tflite_model = converter.convert()
with open('model/model.tflite', 'wb') as f:
  f.write(tflite_model)

print(len(class_names))
for i in range(0, 30):
    target = image_batch[i].numpy()
    # make it to cv2 format
    target = target[...,::-1]

    result = label_image.inference(target)
    if (result != predicted_label_batch[n].title()):
        print(f"{result} and {predicted_label_batch[n].title()} Not matching!")
