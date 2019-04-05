import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

img_path = '630.jpg'
image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    plt.imshow(img_data.eval())
    plt.show()


    resize1 = tf.image.resize_images(img_data, [32, 32], method=0)
    resize1 = np.asarray(resize1.eval(), dtype='uint8')
    plt.imshow(resize1)
    plt.show()

    resize2 = tf.image.resize_images(img_data, [32, 32], method=1)
    resize2 = np.asarray(resize2.eval(), dtype='uint8')
    plt.imshow(resize2)
    plt.show()

    resize3 = tf.image.resize_images(img_data, [32, 32], method=2)
    resize3 = np.asarray(resize3.eval(), dtype='uint8')
    plt.imshow(resize3)
    plt.show()

    resize4 = tf.image.resize_images(img_data, [32, 32], method=3)
    resize4 = np.asarray(resize4.eval(), dtype='uint8')
    plt.imshow(resize4)
    plt.show()

    croped = tf.image.resize_image_with_crop_or_pad(img_data, 32, 32)
    plt.imshow(croped.eval())
    plt.show()

    central_cropped = tf.image.central_crop(img_data, 0.5)
    central_cropped = tf.image.resize_images(central_cropped, [32, 32], method=0)
    central_cropped = np.asarray(central_cropped.eval(), dtype='uint8')
    plt.imshow(central_cropped)
    plt.show()

