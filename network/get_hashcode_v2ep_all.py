import tensorflow as tf
import numpy as np
import os
import sys
#import dsh_dishNet

train_dir = '../data/train_data/'
test_dir = '../data/test_data/'
model_dir = './model%s/' % (sys.argv[1])
batch_size = 100
output_dir = '../data/output%s/' % (sys.argv[1])
#model_dir_runtime = '/root/lsy/dishRetrieval/network/model_web/'
model_dir_runtime = '/Users/lansy/Desktop/graduateDesign/dishRetrieval/network/model_web/'
img_size = 32



#x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
k = 8

def prefix_image(image, resize_w, resize_h):
    image = tf.cast(image, tf.string)
    image_c = tf.read_file(image)
    image = tf.image.decode_jpeg(image_c, channels=3)

    image = tf.image.central_crop(image, 0.5)
    image = tf.image.resize_images(image, [resize_h, resize_w], method=0)
    image = tf.image.per_image_standardization(image)

    image = tf.cast(image, tf.float32)
    with tf.Session() as sess:
        image_numpy = image.eval()
        image_batch = np.array([image_numpy])

    return image_batch


def get_hashcode(image_path):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_dir_runtime + 'model.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_dir_runtime))
        #y_conv = dsh_dishNet.dsh_dish_net(x)
        y_conv = tf.get_collection('y_conv')[0]
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name("input_image/input_image").outputs[0]
        keep_prob = graph.get_tensor_by_name("Placeholder:0")
        image = prefix_image(image_path, img_size, img_size)
        ret = sess.run(y_conv, feed_dict={x: image, keep_prob: 1.0})
        ret1 = tf.reshape(ret, [k])
        #ret2 = sess.run(tf.sign(ret1))
        ret2 = ret1.eval()
        ret_array = [str(i) for i in ret2]
        ret_string = ','.join(ret_array)
    return ret_string


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #already_get = open(output_dir+'train_output.txt', 'r').read()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_dir+'model.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        #y_conv = dsh_dishNet.dsh_dish_net(x)
        y_conv = tf.get_collection('y_conv')[0]
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name("input_image/input_image").outputs[0]
        keep_prob = graph.get_tensor_by_name("Placeholder:0")
        for label in os.listdir(train_dir):
            for pic in os.listdir(train_dir + label):
                image_path = train_dir + label + '/' + pic
                #if image_path in already_get:
                    #continue
                image = prefix_image(image_path, img_size, img_size)
                ret = sess.run(y_conv, feed_dict={x: image, keep_prob: 1.0})
                ret1 = tf.reshape(ret, [k])
                ret2 = sess.run(tf.sign(ret1))
                ret_array1 = [str(i) for i in ret2]
                ret_string1 = ','.join(ret_array1)
                ret3 = ret1.eval()
                ret_array2 = [str(i) for i in ret3]
                ret_string2 = ','.join(ret_array2)
                with open(output_dir+'train_output.txt', 'a') as f1:
                    f1.write("%s\t%s\t%s\t%s\n" % (image_path, label, ret_string1, ret_string2))

        for pic in os.listdir(test_dir):
            image_path = test_dir + pic
            image = prefix_image(image_path, img_size, img_size)
            ret = sess.run(y_conv, feed_dict={x: image, keep_prob: 1.0})
            ret1 = tf.reshape(ret, [k])
            ret2 = sess.run(tf.sign(ret1))
            ret_array1 = [str(i) for i in ret2]
            ret3 = ret1.eval()
            ret_array2 = [str(i) for i in ret3]
            ret_string2 = ','.join(ret_array2)
            with open(output_dir+'test_output.txt', 'a') as f1:
                f1.write("%s\t%s\t%s\n" % (image_path, ','.join(ret_array1), ret_string2))


if __name__ == '__main__':
    main()

