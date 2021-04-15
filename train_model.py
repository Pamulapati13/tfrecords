import tensorflow as tf

class Trainer():
    def __init__(self,train_data,test_data = None):
        self.train_data = train_data
        self.test_data = test_data
        self.model = None

    def build_model(self):
        self.model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (100,100,3)),
                                        tf.keras.layers.MaxPooling2D((2,2)),
                                        tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
                                        tf.keras.layers.MaxPooling2D((2,2)),
                                        tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(64, activation = 'relu'),
                                        tf.keras.layers.Dropout(0.3),
                                        tf.keras.layers.Dense(6)])

    def train(self,epochs):
        self.build_model()
        epochs = epochs
        lr = 1e-3
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr,decay_steps=100,decay_rate=0.96,staircase=True)
        self.opt = tf.keras.optimizers.Adam(self.lr_schedule)
        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        #Custom training loop to train the model
        print('Training Model.....')
        for i in range(epochs):
            acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy()
            loss_tracker = tf.keras.metrics.Mean()
            for data in self.train_data:
                with tf.GradientTape() as t:
                    ip = data[0]
                    op = data[1]
                    preds = self.model(ip)
                    loss = self.cls_loss(op, preds)

                    grads = t.gradient(loss, self.model.trainable_variables)
                    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
                    loss_tracker.update_state(loss)
                    acc_tracker.update_state(op, preds)
            print("epoch : {}, loss : {}, accuracy : {}".format(i, loss_tracker.result(), acc_tracker.result()))
        Print('End of training')

    def test_model(self):
        acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy()
        for data in self.test_data:
            ip = data[0]
            op = data[1]
            preds = self.model(ip)
            acc_tracker.update_state(op, preds)
        print('The test set accuracy is : {}%'.format(acc_tracker.result()))
