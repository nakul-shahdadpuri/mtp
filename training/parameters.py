from training import train_model

epochs = 10
batch_size = 8
k_init = tf.keras.initializers.random_normal(stddev=0.008, seed = 101)      
regularizer = tf.keras.regularizers.L2(1e-4)
b_init = tf.constant_initializer()

train_data, val_data = data_path(orig_img_path = '../input/dehaze/clear_images', hazy_img_path = '../input/dehaze/haze')
train, val = dataloader(train_data, val_data, batch_size)

optimizer = Adam(learning_rate = 1e-4)    # we are using Adam optimizer.
net = gman_net()

train_loss_tracker = tf.keras.metrics.MeanSquaredError(name = "train loss")    # We are using MSE as loss metrics.
val_loss_tracker = tf.keras.metrics.MeanSquaredError(name = "val loss")

# Call the training function.
train_model(epochs, train, val, net, train_loss_tracker, val_loss_tracker, optimizer)
