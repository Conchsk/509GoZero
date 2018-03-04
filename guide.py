import tensorflow as tf


def func_constant():
    a = tf.constant([[1, 2]])
    b = tf.constant([[1], [2]])
    c = tf.matmul(a, b)
    sess = tf.Session()
    print(sess.run(c))
    sess.close()


def func_variable():
    state = tf.Variable(0, name="counter")
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
    sess.close()


def func_fetch():
    input1 = tf.constant(1)
    input2 = tf.constant(2)
    input3 = tf.constant(3)
    intermedia = tf.add(input1, input2)
    result = tf.multiply(intermedia, input3)
    sess = tf.Session()
    print(sess.run([intermedia, result]))
    sess.close()


def func_feed():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    c = tf.add(a, b)
    sess = tf.Session()
    print(sess.run(c, feed_dict={a: [1], b: [2]}))
    sess.close()


def func_xor():
    x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_data = [[0], [1], [1], [0]]
    x = tf.placeholder("float", shape=[None, 2])
    y = tf.placeholder("float", shape=[None, 1])
    weights = {
        "w1": tf.Variable(tf.random_normal([2, 2])),
        "w2": tf.Variable(tf.random_normal([2, 1]))
    }
    biases = {
        "b1": tf.Variable(tf.random_normal([1, 2])),
        "b2": tf.constant(0.0)
    }

    h1 = tf.matmul(x, weights["w1"]) + biases["b1"]
    a1 = tf.nn.relu(h1)
    h2 = tf.matmul(a1, weights["w2"]) + biases["b2"]
    a2 = tf.nn.relu(h2)
    y_pre = a2

    mse = tf.reduce_mean(tf.square(y_pre - y))
    train_step = tf.train.AdamOptimizer(0.01).minimize(mse)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(100000):
        x_batch = x_data
        y_batch = y_data
        sess.run(train_step, feed_dict={x: x_batch, y: y_batch})

    print(f'{sess.run([y_pre], feed_dict={x: x_data, y: y_data})}\n')
    sess.close()


if __name__ == '__main__':
    # func_constant()
    # func_variable()
    # func_fetch()
    # func_feed()
    func_xor()
