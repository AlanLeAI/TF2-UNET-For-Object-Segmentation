import tensorflow as tf
from tensorflow.keras import layers

class DoubleConv(tf.keras.Model):
    def __init__(self, out_channels):
        super(DoubleConv,self).__init__()
        self.conv = tf.keras.models.Sequential([
            layers.Conv2D(out_channels, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(out_channels, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ])

    def call(self, inputs):
        return self.conv(inputs)

class UNET(tf.keras.Model):
    def __init__(self, in_channels=3, out_channels =1, features = [64,128, 256, 512]):
        super(UNET, self).__init__()
        self.downs = []
        self.ups = []
        self.pool = layers.MaxPooling2D(pool_size=(2,2),strides=2)

        #Down Part
        for feature in features:
            self.downs.append(DoubleConv(feature))

        #UP Part
        for feature in reversed(features):
            self.ups.append(
                layers.Conv2DTranspose(feature, kernel_size=2, strides=2)
            )
            self.ups.append(
                DoubleConv(feature)
            )
        
        self.bottleneck = DoubleConv(features[-1]*2)
        self.final_conv = layers.Conv2D(out_channels, kernel_size= 1)
    
    def call(self, inputs):

        skip_connection = []
        for down in self.downs:
            inputs = down(inputs)
            skip_connection.append(inputs)
            inputs = self.pool(inputs)
        
        x = self.bottleneck(inputs)
        skip_connection = skip_connection[::-1]

        for idx in range(0, len(self.ups),2):
            x = self.ups[idx](x)
            skip = skip_connection[idx//2]

            concat_skip = tf.concat((skip,x), axis=3)
            x= self.ups[idx+1](concat_skip)
        return self.final_conv(x)
    
    def model(self):
        x = tf.keras.layers.Input(shape=(160, 160, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

def test():
    init = tf.keras.initializers.GlorotUniform()
    inputs = init([1,160,160,3])
    model = UNET(in_channels=3, out_channels=2)
    outputs = model(inputs)
    model.model().summary()

if __name__=='__main__':
    test()
