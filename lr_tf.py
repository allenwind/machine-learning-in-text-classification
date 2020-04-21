import numpy as np
import tensorflow as tf

class LogisticRegressionLayer(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 1),
                                  initializer='random_normal',
                                  trainable=True)
        self.b = self.add_weight(shape=(1,),
                                  initializer='random_normal',
                                  trainable=True)

    def call(self, inputs):
        # 仿射变换
        r = tf.matmul(inputs, self.w) + self.b
        # sigmoid 激活输出
        p = tf.math.sigmoid(r)
        return p


inputs = tf.keras.layers.Input(shape=(5,))
lr_layer = LogisticRegressionLayer()(inputs)
model = tf.keras.Model(inputs, lr_layer)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)
model.fit(X, y, epochs=100)
