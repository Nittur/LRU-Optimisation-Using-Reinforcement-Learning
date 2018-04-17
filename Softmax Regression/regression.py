import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt


# number of features
num_features = 2
# number of target labels
num_labels = 5
# learning rate (alpha)
learning_rate = 0.03
# number of epochs
epochs = 100
#batch_size
batch_size = 5


#input data
X = tf.placeholder(tf.float32, shape = (1, num_features))
Y = tf.placeholder(tf.float32, shape = (1, num_labels))


# Variables
weigths =  tf.Variable(tf.truncated_normal([num_features, num_labels]))
biases = tf.Variable(tf.zeros([1, num_labels]))

#Training Computation
logits = tf.matmul(X, weigths) +  biases

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()


validation_logits = tf.nn.softmax(tf.matmul(X, weigths) +  biases)

with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        f = open("sample_training.txt", "r")
        batch_count = 1
        train_loss = 0
        for line in f:
            cont = line[:-1].split(",")
            train_X = np.array([[float(cont[3]), float(cont[4])]])
            train_Y = np.zeros([1, num_labels], dtype = float)
            train_Y[0][int(cont[1])-1] = 1.0
            _, t = sess.run([optimizer, loss], feed_dict={ X:train_X, Y:train_Y })
            batch_count = batch_count + 1
            if batch_count == 81:
                train_loss = t   
        plt.plot(i,train_loss, linestyle='-', marker='o')
        f.close()
    plt.show()


    # Testing
    
    f1 = open("sample_testing.txt", "r")
    arr = []
    for line in f1:
        cont = line[:-1].split(",")
        test_X = np.array([[float(cont[3]), float(cont[4])]])
        test_Y = np.zeros([1, num_labels], dtype = float)
        test_Y[0][int(cont[1])-1] = 1.0
        test_set = sess.run(validation_logits, feed_dict={X:test_X, Y:test_Y})
        arr.append(np.argmax(test_set) == (int(cont[1]) - 1))
    print("Accuracy", np.sum(arr)*100/51)    
    f1.close()
    
    







           
        

      