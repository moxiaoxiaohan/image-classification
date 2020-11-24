from __future__ import absolute_import

from __future__ import division

from __future__ import print_function
import math
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


def conv_bn_layer(input,

                      num_filters,

                      filter_size,

                      stride=1,

                      groups=1,

                      act=None,

                      name=None):

        conv = fluid.layers.conv2d(

            input=input,

            num_filters=num_filters,

            filter_size=filter_size,

            stride=stride,

            padding=(filter_size - 1) // 2,

            groups=groups,

            act=None,

            param_attr=ParamAttr(name=name + "_weights"),

            bias_attr=False,

            name=name + '.conv2d.output.1')

        if name == "conv1":

            bn_name = "bn_" + name

        else:

            bn_name = "bn" + name[3:]

        return fluid.layers.batch_norm(

            input=conv,

            act=act,

            name=bn_name + '.output.1',

            param_attr=ParamAttr(name=bn_name + '_scale'),

            bias_attr=ParamAttr(bn_name + '_offset'),

            moving_mean_name=bn_name + '_mean',

            moving_variance_name=bn_name + '_variance', )

def shortcut(input, ch_out, stride, name):

        ch_in = input.shape[1]

        if ch_in != ch_out or stride != 1:

            return conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input
def bottleneck_block( input, num_filters, stride, cardinality, name):

        cardinality = 32

        conv0 = conv_bn_layer(

            input=input,

            num_filters=num_filters,

            filter_size=1,

            act='relu',

            name=name + "_branch2a")

        conv1 =conv_bn_layer(

            input=conv0,

            num_filters=num_filters,

            filter_size=3,

            stride=stride,

            groups=cardinality,

            act='relu',

            name=name + "_branch2b")

        conv2 = conv_bn_layer(input=conv1,

            num_filters=num_filters if cardinality == 64 else num_filters * 2,

            filter_size=1,

            act=None,

            name=name + "_branch2c")

        short = shortcut(input,

            num_filters if cardinality == 64 else num_filters * 2,

            stride,

            name=name + "_branch1")

        return fluid.layers.elementwise_add(

            x=short, y=conv2, act='relu', name=name + ".add.output.5")
def net(input,class_dim):

        layers = 50

        cardinality = 32

        supported_layers = [50, 101, 152]

        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:

            depth = [3, 4, 6, 3]

        elif layers == 101:

            depth = [3, 4, 23, 3]

        elif layers == 152:

            depth = [3, 8, 36, 3]

        num_filters1 = [256, 512, 1024, 2048]

        num_filters2 = [128, 256, 512, 1024]

        conv = conv_bn_layer(input=input,

            num_filters=64,

            filter_size=7,

            stride=2,

            act='relu',

            name="res_conv1")  #debug

        conv = fluid.layers.pool2d(

            input=conv,

            pool_size=3,

            pool_stride=2,

            pool_padding=1,

            pool_type='max')

        for block in range(len(depth)):

            for i in range(depth[block]):

                if layers in [101, 152] and block == 2:

                    if i == 0:

                        conv_name = "res" + str(block + 2) + "a"

                    else:

                        conv_name = "res" + str(block + 2) + "b" + str(i)

                else:

                    conv_name = "res" + str(block + 2) + chr(97 + i)

                conv = bottleneck_block(

                    input=conv,

                    num_filters=num_filters1[block]

                    if cardinality == 64 else num_filters2[block],
                    
                    stride=2 if i == 0 and block != 0 else 1,

                    cardinality=cardinality,

                    name=conv_name)

        pool = fluid.layers.pool2d(

            input=conv, pool_type='avg', global_pooling=True)
        #drop = fluid.layers.dropout(x=pool, dropout_prob=0.5)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        
        out = fluid.layers.fc(input=pool,size=class_dim,act='softmax',
                                param_attr=fluid.param_attr.ParamAttr(

                                initializer=fluid.initializer.Uniform(-stdv, stdv),

                                name='fc_weights'),

                                bias_attr=fluid.param_attr.ParamAttr(name='fc_offset'))
        

        return out
