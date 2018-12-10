import tensorflow as tf
import tensorflow.contrib.slim as slim


class MobileNet(object):
    def __init__(self, input, num_classes=1000, weight_decay=0.0004, is_training=True):
        self.imgs      = input
        self.width_multiplier = 1.0
        self.resolution_multiplier = 1.0
        self.num_classes = num_classes
        self.decay = weight_decay
        self.is_training = is_training
        self.end_points = {}
        self.build_model()

    def build_model(self):
        with tf.variable_scope('Mobilenet') as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                                weights_initializer=slim.initializers.xavier_initializer(),
                                biases_initializer=slim.init_ops.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(self.decay),
                                activation_fn=None,
                                outputs_collections=[end_points_collection]):
                with slim.arg_scope([slim.batch_norm],
                                    is_training=self.is_training,
                                    activation_fn=tf.nn.relu):
                    net = slim.convolution2d(self.imgs, round(32 * self.width_multiplier), [3, 3], stride=2, padding='SAME',
                                             scope='conv_1')
                    net = slim.batch_norm(net, scope='conv_1/batch_norm')
                    net = self._depthwise_separable_conv(net, 64, sc='conv_ds_2')
                    net = self._depthwise_separable_conv(net, 128, downsample=True, sc='conv_ds_3')
                    net = self._depthwise_separable_conv(net, 128, sc='conv_ds_4')
                    net = self._depthwise_separable_conv(net, 256, downsample=True, sc='conv_ds_5')
                    net = self._depthwise_separable_conv(net, 256, sc='conv_ds_6')
                    net = self._depthwise_separable_conv(net, 512, downsample=True, sc='conv_ds_7')

                    net = self._depthwise_separable_conv(net, 512, sc='conv_ds_8')
                    net = self._depthwise_separable_conv(net, 512, sc='conv_ds_9')
                    net = self._depthwise_separable_conv(net, 512, sc='conv_ds_10')
                    net = self._depthwise_separable_conv(net, 512, sc='conv_ds_11')
                    net = self._depthwise_separable_conv(net, 512, sc='conv_ds_12')

                    net = self._depthwise_separable_conv(net, 1024, downsample=True, sc='conv_ds_13')
                    net = self._depthwise_separable_conv(net, 1024, sc='conv_ds_14')
                    net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            end_points['squeeze'] = net
            logits = slim.fully_connected(net, self.num_classes, activation_fn=None, scope='Logists/Conv2d_1c_1x1')
            predictions = slim.softmax(logits, scope='Predictions')

            end_points['logits'] = logits
            end_points['predictions'] = predictions
            self.end_points = end_points

    def _depthwise_separable_conv(self, inputs,
                                  num_pwc_filters,
                                  sc,
                                  downsample=False):
        """ Helper function to build the depth-wise separable convolution layer.
        """
        num_pwc_filters = round(num_pwc_filters * self.width_multiplier)
        _stride = 2 if downsample else 1

        # skip pointwise by setting num_outputs=None
        dw_sc = sc+'_depthwise'
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=_stride,
                                                      depth_multiplier=1,
                                                      kernel_size=[3, 3],
                                                      scope=dw_sc)

        bn = slim.batch_norm(depthwise_conv, scope=dw_sc + '/BatchNorm')
        pw_sc = sc+'_pointwise'
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=pw_sc)
        bn = slim.batch_norm(pointwise_conv, scope=pw_sc + '/BatchNorm')
        return bn

    def add_summary(self, x):
        tf.summary.histogram(x.op.name+'/activations', x)
        tf.summary.scalar(x.op.name+'/sparsity', tf.nn.zero_fraction(x))




