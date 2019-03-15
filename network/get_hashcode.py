import tensorflow as tf
import os
import dsh_dishNet

train_dir = '../data/train_data/'
test_dir = '../data/test_data/'
model_dir = './model/'
batch_size = 100
output_dir = '../data/output/'


x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
k = 12

def prefix_image(image, resize_w, resize_h):
    image = tf.cast(image, tf.string)
    image_c = tf.read_file(image)
    image = tf.image.decode_jpeg(image_c, channels=3)

    image = tf.image.resize_images(image, [resize_h, resize_w], method=0)
    image = tf.image.per_image_standardization(image)

    image = tf.cast(image, tf.float32)

    return image

def main():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_dir+'model.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        y_conv = dsh_dishNet.dsh_dish_net(x)
        for label in os.listdir(train_dir):
            for pic in os.listdir(train_dir + label):
                image_path = train_dir + label + '/' + pic
                image = prefix_image(image_path, 32, 32)
                ret = sess.run(y_conv, feed_dict={x: image})
                ret1 = tf.reshape(ret, [k])
                ret2 = sess.run(tf.sign(ret1))
                ret_array = ret2.eval()
                ret_string = ','.join(ret_array)
                with open(output_dir+'train_output', 'a') as f1:
                    f1.write("%s\t%s\n" % (image_path, ret_string))

        for pic in os.listdir(test_dir):
            image_path = test_dir + pic
            image = prefix_image(image_path, 32, 32)
            ret = sess.run(y_conv, feed_dict={x: image})
            ret1 = tf.reshape(ret, [k])
            ret2 = sess.run(tf.sign(ret1))
            ret_array = ret2.eval()
            ret_string = ','.join(ret_array)
            with open(output_dir+'test_output', 'a') as f1:
                f1.write("%s\t%s\n" % (image_path, ret_string))


if __name__ == '__main__':
    main()



