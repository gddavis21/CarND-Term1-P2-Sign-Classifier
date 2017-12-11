# Load pickled data
import numpy as np
import tensorflow as tf
import cv2
import os
import csv
import pickle
import math
import random
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

# TODO: Fill this in based on where you saved the training and testing data

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def preprocess_image(img):
    mean = np.mean(img, axis=(0,1))
    stdev = np.std(img, axis=(0,1))
    numPixels = img.shape[0] * img.shape[1]
    adj_stdev = np.maximum(stdev, math.sqrt(1.0/numPixels)) 
    return (img.astype(np.float32) - mean) / adj_stdev
    
def preprocess_images(images):
    return np.array([preprocess_image(img) for img in images])
    
def crop_training_image(img):
    y = random.randint(0,8)
    x = random.randint(0,8)
    return img[y:y+24, x:x+24, :]
    
def crop_training_images(imgs):
    return np.array([crop_training_image(img) for img in imgs])
    
def crop_test_image(img):
    return img[4:28, 4:28, :]
    
def crop_test_images(imgs):
    return np.array([crop_test_image(img) for img in imgs])
    
def load_sign_data(file_path):
    '''
    '''
    with open(file_path, mode='rb') as f:
        data = pickle.load(f)
        return data['features'], data['labels']
        
def load_sign_names(file_path):
    '''
    '''
    sign_names = {}
    with open(file_path, newline='') as f:
        names_reader = csv.reader(f)
        next(names_reader)  # skip header
        for row in names_reader:
            sign_names[int(row[0])] = row[1]
    return sign_names
    
def my_conv2d(name, input, weights_shape, weight_decay, strides, padding):
    '''
    '''
    fan_in = weights_shape[0] * weights_shape[1] * weights_shape[2]
    sigma = (2.0/fan_in)**0.5
    with tf.variable_scope(name):
        w = tf.get_variable(
            name='weights', 
            shape=weights_shape, 
            initializer=tf.random_normal_initializer(stddev=sigma))
        if weight_decay is not None:
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(w), weight_decay))
        b = tf.get_variable(
            name='biases',
            initializer=tf.zeros([weights_shape[3]]))
        conv = tf.nn.conv2d(input, w, strides=strides, padding=padding)
        return tf.nn.relu(tf.nn.bias_add(conv, b))
    
# def my_max_pool(input, k, padding):
    # return tf.nn.max_pool(input, ksize=[1,k,k,1], strides=[1,k,k,1], padding=padding)
    
def my_fully_conn_nonlinear(name, input, weights_shape, weight_decay):
    '''
    '''
    fan_in = weights_shape[0]
    sigma = (2.0/fan_in)**0.5
    with tf.variable_scope(name):
        w = tf.get_variable(
            name='weights', 
            shape=weights_shape, 
            initializer=tf.random_normal_initializer(stddev=sigma))
        if weight_decay is not None:
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(w), weight_decay))
        b = tf.get_variable(
            name='biases',
            initializer=tf.zeros([weights_shape[1]]))
        return tf.nn.relu(tf.add(tf.matmul(input, w), b))
    
def my_fully_conn_linear(name, input, weights_shape, weight_decay):
    '''
    '''
    fan_in = weights_shape[0]
    sigma = (2.0/fan_in)**0.5
    with tf.variable_scope(name):
        w = tf.get_variable(
            name='weights', 
            shape=weights_shape, 
            initializer=tf.random_normal_initializer(stddev=sigma))
        if weight_decay is not None:
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(w), weight_decay))
        b = tf.get_variable(
            name='biases',
            initializer=tf.zeros([weights_shape[1]]))
        return tf.add(tf.matmul(input, w), b)
    
def inference(images, num_classes, weight_decay):    
    '''
    '''
    x_image = tf.reshape(images, [-1, 24, 24, 3])
    keep_prob = tf.placeholder(tf.float32)
    
    # Layer 1: Convolutional. Input = 24x24x3. Output = 24x24x32.
    conv_1 = my_conv2d(
        name='conv-1', 
        input=x_image, 
        weights_shape=[5,5,3,32], 
        weight_decay=weight_decay,
        strides=[1,1,1,1], 
        padding='SAME')
    print(conv_1.shape)

    # Pooling. Input = 24x24x32. Output = 12x12x32.
    pool_1 = tf.nn.max_pool(
        conv_1, 
        ksize=[1,2,2,1], 
        strides=[1,2,2,1], 
        padding='VALID')
    print(pool_1.shape)
    
    # Layer 2: Convolutional. Input = 12x12x32. Output = 12x12x64.
    conv_2 = my_conv2d(
        name='conv-2', 
        input=pool_1, 
        weights_shape=[5,5,32,64], 
        weight_decay=weight_decay,
        strides=[1,1,1,1], 
        padding='SAME')
    print(conv_2.shape)

    # Pooling. Input = 12x12x64. Output = 6x6x64.
    pool_2 = tf.nn.max_pool(
        conv_2, 
        ksize=[1,2,2,1], 
        strides=[1,2,2,1], 
        padding='VALID')
    print(pool_2.shape)
    
    # Flatten. Input = 6x6x64. Output = 2304.
    flat_2 = flatten(pool_2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc_3 = my_fully_conn_nonlinear(
        name='fc-3', 
        input=tf.nn.dropout(flat_2, keep_prob), 
        # input=flat_2, 
        weights_shape=[6*6*64,1024],
        weight_decay=weight_decay)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc_4 = my_fully_conn_nonlinear(
        name='fc-4', 
        input=tf.nn.dropout(fc_3, keep_prob), 
        # input=fc_3, 
        weights_shape=[1024,512],
        weight_decay=weight_decay)
    
    fc_5 = my_fully_conn_nonlinear(
        name='fc-5', 
        input=tf.nn.dropout(fc_4, keep_prob), 
        # input=fc_3, 
        weights_shape=[512,256],
        weight_decay=weight_decay)
    
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    # dropout before final readout
    logits = my_fully_conn_linear(
        name='readout', 
        input=tf.nn.dropout(fc_5, keep_prob), 
        weights_shape=[256,num_classes],
        weight_decay=weight_decay)
        
    return logits, keep_prob
    
def loss(logits, labels):
    '''
    '''
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    mean_xentropy = tf.reduce_mean(xentropy)
    tf.add_to_collection('losses', mean_xentropy)
    return tf.add_n(tf.get_collection('losses'))
    
def training(loss_op, learn_rate):
    '''
    '''
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    return optimizer.minimize(loss_op)

def accuracy(logits, labels):
    '''
    '''
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def do_eval(
    sess, loss_op, accuracy_op, x, y, keep_prob,
    X_data, y_data, batch_size):
    '''
    '''
    total_loss = 0
    total_accuracy = 0
    for offset in range(0, len(X_data), batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        loss, accuracy = sess.run(
            [loss_op, accuracy_op], 
            feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_loss += loss
        total_accuracy += (accuracy * len(batch_x))
    return total_loss, total_accuracy / len(X_data)

# ============= SCRIPT ========================================================

X_train, y_train = load_sign_data('../traffic-sign-data/train.p')
X_valid, y_valid = load_sign_data('../traffic-sign-data/valid.p')
X_test, y_test = load_sign_data('../traffic-sign-data/test.p')

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
sign_names = load_sign_names('signnames.csv')
n_classes = len(sign_names)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

X_train = preprocess_images(X_train)
X_valid = preprocess_images(X_valid)
X_test = preprocess_images(X_test)

X_valid = crop_test_images(X_valid)
X_test = crop_test_images(X_test)

NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARN_RATE = 0.0003
WEIGHT_DECAY_RATE = None

# classifier model
x = tf.placeholder(tf.float32, (None, 24, 24, 3))
y = tf.placeholder(tf.int32, (None))
logits, keep_prob = inference(x, n_classes, WEIGHT_DECAY_RATE)
labels = tf.one_hot(y, n_classes)

# training
loss_op = loss(logits, labels)
train_op = training(loss_op, LEARN_RATE)

# evaluation
accuracy_op = accuracy(logits, labels)

# saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    print("Training...")
        
    for i in range(NUM_EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        X_train_cropped = crop_training_images(X_train)

        for offset in range(0, len(X_train), BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_cropped[offset:end], y_train[offset:end]
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.4})
            
        train_loss, train_accuracy = do_eval(
            sess, loss_op, accuracy_op, x, y, keep_prob,
            X_train_cropped, y_train, BATCH_SIZE)
            
        _, valid_accuracy = do_eval(
            sess, loss_op, accuracy_op, x, y, keep_prob,
            X_valid, y_valid, BATCH_SIZE)
        
        print('Epoch %d: loss = %.2f, training accuracy = %.3f, validation accuracy = %.3f' % (i+1, train_loss, train_accuracy, valid_accuracy))
        
    # saver.save(sess, './lenet')
    # print("Model saved")