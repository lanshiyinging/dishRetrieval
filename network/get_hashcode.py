import tensorflow as tf
import numpy as np
import os
#import dsh_dishNet

train_dir = '../data/train_data/'
test_dir = '../data/test_data/'
model_dir = './model/'
batch_size = 100
output_dir = '../data/output/'
model_dir_runtime = '/root/lsy/dishRetrieval/network/model/'



#x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
k = 12

def prefix_image(image, resize_w, resize_h):
    image = tf.cast(image, tf.string)
    image_c = tf.read_file(image)
    image = tf.image.decode_jpeg(image_c, channels=3)

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
        image = prefix_image(image_path, 32, 32)
        ret = sess.run(y_conv, feed_dict={x: image})
        ret1 = tf.reshape(ret, [k])
        ret2 = sess.run(tf.sign(ret1))
        #ret_array = ret2.eval()
        #ret_array = [str(i) for i in ret2]
        ret_string = ','.join(ret2)
    return ret_string


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    already_get = open(output_dir+'train_output.txt', 'r').read()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_dir+'model.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        #y_conv = dsh_dishNet.dsh_dish_net(x)
        y_conv = tf.get_collection('y_conv')[0]
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name("input_image/input_image").outputs[0]
        for label in os.listdir(train_dir):
            for pic in os.listdir(train_dir + label):
                image_path = train_dir + label + '/' + pic
                if image_path in already_get:
                    continue
                image = prefix_image(image_path, 32, 32)
                ret = sess.run(y_conv, feed_dict={x: image})
                ret1 = tf.reshape(ret, [k])
                ret2 = sess.run(tf.sign(ret1))
                #ret_array = ret2.eval()
                #ret2 = ret2.astype('str')
                ret_array = [str(i) for i in ret2]
                ret_string = ','.join(ret_array)
                with open(output_dir+'train_output.txt', 'a') as f1:
                    f1.write("%s\t%s\t%s\n" % (image_path, label, ret_string))

        for pic in os.listdir(test_dir):
            image_path = test_dir + pic
            image = prefix_image(image_path, 32, 32)
            ret = sess.run(y_conv, feed_dict={x: image})
            ret1 = tf.reshape(ret, [k])
            ret2 = sess.run(tf.sign(ret1))
            #ret_array = ret2.eval()
            ret_array = [str(i) for i in ret2]
            ret_string = ','.join(ret_array)
            with open(output_dir+'test_output.txt', 'a') as f1:
                f1.write("%s\t%s\n" % (image_path, ret_string))


if __name__ == '__main__':
    main()

