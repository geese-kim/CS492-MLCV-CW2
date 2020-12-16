import tensorflow as tf

def data_preprocessing(bufsize, batsize):

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # (60000, 28, 28) (60000,), (10000, 28, 28) (10000,)

    ## The input values whose range is [0, 255] will be normalized [-1, 1]
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    ## The input values whose range is [0, 255] will be normalized [-1, 1]
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    test_images = (test_images - 127.5) / 127.5

    ## Create data batch and shuffle
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images) \
                                    .shuffle(bufsize) \
                                    .batch(batsize)
    
    return train_dataset, train_labels, test_images, test_labels 