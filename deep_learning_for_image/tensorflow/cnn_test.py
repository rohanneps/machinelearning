import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


data = input_data.read_data_sets('data/fashion',one_hot=True)

train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1,28,28,1)

train_y = data.train.labels
test_y = data.test.labels

training_iters = 5 
learning_rate = 0.001 
batch_size = 128

x = tf.placeholder(dtype=tf.float32,shape=[None,28,28,1])
y = tf.placeholder(dtype=tf.float32,shape=[None,10])



def conv2d(x,W,b,strides=1):
	x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1], padding='SAME')
	x = tf.nn.bias_add(x,b)
	return tf.nn.relu(x)


def maxpool2d(x,k=2):
	return tf.nn.max_pool(x,ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,10), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}

def conv_net(x, weights, biases):
    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)
    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

# with tf.Session() as sess:

# Training Phase
sess = tf.Session()
sess.run(init) 
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
summary_writer = tf.summary.FileWriter('./Output', sess.graph)

saver = tf.train.Saver()

for i in range(training_iters):
    for batch in range(len(train_X)//batch_size):
        batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
        batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
        # Run optimization op (backprop).
            # Calculate batch loss and accuracy
        opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                          y: batch_y})
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                          y: batch_y})
    print("Iter " + str(i) + ", Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    print("Optimization Finished!")
    # Calculate accuracy for all 10000 mnist test images
    test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
    train_loss.append(loss)
    test_loss.append(valid_loss)
    train_accuracy.append(acc)
    test_accuracy.append(test_acc)
    print("Testing Accuracy:","{:.5f}".format(test_acc))
summary_writer.close()
save_path = saver.save(sess, "./model.ckpt")


# # Making Prediction
xt = test_X[0]
xt = np.expand_dims(xt,0)

p = sess.run(pred,feed_dict={x:xt})[0]
print('Label is {}'.format(np.argmax(p))