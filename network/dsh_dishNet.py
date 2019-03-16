import tensorflow as tf
import numpy as np
import os
import time

config = tf.ConfigProto(log_device_placement=True,
                        inter_op_parallelism_threads=4,
                        intra_op_parallelism_threads=4,
                        allow_soft_placement=True)


k = 12
batch_size = 100
epoch_num = 1
momentum = 0.9
weight_decay = 0.004
base_lr = 0.001
m = 2 * k
alpha = 0.01

with tf.name_scope("input_image"):
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    tf.summary.image('input_image', x, 3)
y = tf.placeholder(tf.float32, shape=[batch_size])



def get_files(filename):
    num = 0
    train_image = []
    train_label = []
    for label in os.listdir(filename):
        for pic in os.listdir(filename + label):
            train_image.append(filename + label + '/' + pic)
            train_label.append(label)
            num += 1
    temp = np.array([train_image, train_label])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list, num


def get_batches(image, label, resize_w, resize_h, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)
    queue = tf.train.slice_input_producer([image, label], num_epochs=epoch_num)
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c, channels=3)

    image = tf.image.resize_images(image, [resize_h, resize_w], method=0)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    images_batch = tf.cast(image_batch, tf.float32)
    labels_batch = tf.reshape(label_batch, [batch_size])

    return images_batch, labels_batch


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))

    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))

    tf.summary.histogram('histogram', var)


def weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name=name, shape=shape, initializer=initializer)


def bias_variable(name, shape):
    initializer = tf.constant_initializer(0)
    return tf.get_variable(name=name, shape=shape, initializer=initializer)


def conv_layer(inputs, W, conv_strides, padding):
    return tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)


def max_pool_layer(inputs, kernal_size, pool_strides):
    return tf.nn.max_pool(inputs, ksize=kernal_size, strides=pool_strides, padding='SAME')


def average_pool_layer(inputs, kernal_size, pool_strides):
    return tf.nn.avg_pool(inputs, ksize=kernal_size, strides=pool_strides, padding='SAME')


def dsh_dish_net(inputs):
    inputs_shape = inputs.get_shape()
    inputs = tf.reshape(inputs, shape=[-1, inputs_shape[1].value, inputs_shape[2].value, inputs_shape[3].value])
    with tf.name_scope("dsh_dish_net"):
        with tf.name_scope("layer1"):
            with tf.name_scope("weights"):
                W_conv1 = weight_variable("W_conv1", [5, 5, 3, 32])
                variable_summaries(W_conv1)
            with tf.name_scope("biases"):
                b_conv1 = bias_variable("b_conv1", [32])
                variable_summaries(b_conv1)
            conv_strides = [1, 1, 1, 1]
            kernal_size = [1, 3, 3, 1]
            pool_strides = [1, 2, 2, 1]
            conv1 = conv_layer(inputs, W_conv1, conv_strides, 'SAME')
            pool1 = max_pool_layer(conv1+b_conv1, kernal_size, pool_strides)
            relu1 = tf.nn.relu(pool1)
            norm1 = tf.nn.lrn(relu1, 3, bias=1.0, alpha=5e-05, beta=0.75, name='norm1')

        with tf.name_scope("layer2"):
            with tf.name_scope("weights"):
                W_conv2 = weight_variable("W_conv2", [5, 5, 32, 32])
                variable_summaries(W_conv2)
            with tf.name_scope("biases"):
                b_conv2 = bias_variable("b_conv2", [32])
                variable_summaries(b_conv2)
            conv2 = conv_layer(norm1, W_conv2, conv_strides, 'SAME')
            pool2 = average_pool_layer(conv2+b_conv2, kernal_size, pool_strides)
            relu2 = tf.nn.relu(pool2)
            norm2 = tf.nn.lrn(relu2, 3, bias=1.0, alpha=5e-05, beta=0.75, name='norm2')

        with tf.name_scope("layer3"):
            with tf.name_scope("weights"):
                W_conv3 = weight_variable("W_conv3", [5, 5, 32, 64])
                variable_summaries(W_conv3)
            with tf.name_scope("biases"):
                b_conv3 = bias_variable("b_conv3", [64])
                variable_summaries(b_conv3)
            conv3 = conv_layer(norm2, W_conv3, conv_strides, 'SAME')
            relu3 = tf.nn.relu(conv3+b_conv3)
            pool3 = average_pool_layer(relu3, kernal_size, pool_strides)

        shape = pool3.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1]*shape[-2]*shape[-3]
        else:
            size = shape[1]

        with tf.name_scope("fc_layer1"):
            with tf.name_scope("weights"):
                W_fc1 = weight_variable("W_fc1", [size, 500])
                variable_summaries(W_fc1)
            with tf.name_scope("biases"):
                b_fc1 = bias_variable("b_fc1", [500])
                variable_summaries(b_fc1)
            pool3_flat = tf.reshape(pool3, [-1, size])
            fc1 = tf.nn.relu(tf.matmul(pool3_flat, W_fc1) + b_fc1)

        with tf.name_scope("fc_layer2"):
            with tf.name_scope("weights"):
                W_fc2 = weight_variable("W_fc2", [500, k])
                variable_summaries(W_fc2)
            with tf.name_scope("biases"):
                b_fc2 = bias_variable("b_fc2", [k])
                variable_summaries(b_fc2)
            y_conv = tf.matmul(fc1, W_fc2) + b_fc2

    return y_conv


def loss_function(y_conv, label_batches):
    global m
    num = batch_size
    shape = y_conv.get_shape().as_list()
    print(shape)
    Lr = 0
    Lr = tf.cast(Lr, tf.float32)
    y_conv = tf.cast(y_conv, tf.float32)
    y_conv = tf.transpose(y_conv)
    shape = label_batches.get_shape().as_list()
    print(shape)
    for i in range(num):
        b1 = y_conv[:, i]
        for j in range(i+1, num):
            b2 = y_conv[:, j]
            l2_dis = tf.sqrt(tf.reduce_sum(tf.square(b1-b2)))
            norm = alpha * (tf.subtract(tf.abs(b1), 1.0) + tf.subtract(tf.abs(b2), 1.0))
            m = tf.cast(m, tf.float32)
            Lr = Lr + tf.where(tf.equal(label_batches[i], label_batches[j]), l2_dis/2.0, tf.maximum(tf.subtract(m, l2_dis), 0)/2.0) + norm
    cost = tf.reduce_mean(Lr)
    return cost


def main():
    train_data_dir = "../data/train_data/"
    #val_data_dir = "data/val_data/"
    #test_data_dir = "data/test_data/"
    train_image, train_label, train_num = get_files(train_data_dir)
    train_image_batches, train_label_batches = get_batches(train_image, train_label, 32, 32, batch_size, train_num)

    y_conv = dsh_dish_net(x)
    with tf.name_scope('loss'):
        loss = loss_function(y_conv, y)
        tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=base_lr, global_step=global_step, decay_steps=10, decay_rate=0.4, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter("logs/", sess.graph)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        #iteration = 1 + int(train_num / batch_size)
        start_time = time.time()
        #for iter in range(iteration):
        count = 0
        try:
            while not coord.should_stop():
                batch_images, batch_labels = sess.run([train_image_batches, train_label_batches])
                _, loss_record, result = sess.run([train_step, loss, merged], feed_dict={x: batch_images, y: batch_labels})
                #result = sess.run(merged, feed_dict={x: image_batches, y: label_batches})
                writer.add_summary(result, count)
                #loss_record = sess.run(loss, feed_dict={x: image_batches, y: label_batches})
                end_time = time.time()
                duration = end_time - start_time
                print("iteration:%d\tloss:%f\tduration:%s\n" % (count, loss_record, duration))
                start_time = end_time
                print("------------iteration %d is finished---------" % count)
                count += 1
        except tf.errors.OutOfRangeError:
            print("Done!")
        finally:
            coord.request_stop()
            print("All threads are asked to stop!")
        coord.join(threads)
        if not os.path.exists("./model/"):
            os.makedirs("./model/")
        saver.save(sess, "./model/")
        print("Optimization Finished!")

if __name__ == '__main__':
    main()



