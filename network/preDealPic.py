import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

img_path = '0.jpg'
image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    plt.imshow(img_data.eval())
    plt.show()

    central_cropped = tf.image.central_crop(img_data, 0.5)
    plt.imshow(central_cropped.eval())
    plt.show()

    central_cropped1 = tf.image.resize_images(central_cropped, [32, 32], method=0)
    central_cropped1 = np.asarray(central_cropped1.eval(), dtype='uint8')
    plt.imshow(central_cropped1)
    plt.show()

    central_cropped2 = tf.image.resize_images(central_cropped, [32, 32], method=1)
    central_cropped2 = np.asarray(central_cropped2.eval(), dtype='uint8')
    plt.imshow(central_cropped2)
    plt.show()

    central_cropped3 = tf.image.resize_images(central_cropped, [32, 32], method=2)
    central_cropped3 = np.asarray(central_cropped3.eval(), dtype='uint8')
    plt.imshow(central_cropped3)
    plt.show()

    central_cropped4 = tf.image.resize_images(central_cropped, [32, 32], method=3)
    central_cropped4 = np.asarray(central_cropped4.eval(), dtype='uint8')
    plt.imshow(central_cropped4)
    plt.show()



    '''
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 32, 32)
    plt.imshow(croped.eval())
    plt.show()
    '''



