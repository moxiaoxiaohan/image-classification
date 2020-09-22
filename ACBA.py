from __future__ import absolute_import

from __future__ import division

from __future__ import print_function
import math
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


def conv_bn_layer(input,

                      num_filters,

                      filter_size=1,
                      
                      padding=0,
                    
                      stride=1,
                      dilation=1,

                      groups=1,

                      act=None,

                      name=None):

        conv = fluid.layers.conv2d(

            input=input,

            num_filters=num_filters,

            filter_size=filter_size,

            stride=stride,
            #filter_size - 1) // 2
            #padding=((filter_size - 1) // 2,(filter_size - 1) // 2),
            padding=padding,
            
            groups=groups,

            act=None,

            param_attr=ParamAttr(name=name + "_weights"),

            bias_attr=False,

            name=name + '.conv2d.output.1')
        #print(conv.shape)
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
        #print(input.shape)
        if ch_in != ch_out or stride != 1:
            #print(conv_bn_layer(input, ch_out, 1, 0,stride, name=name).shape)
            return conv_bn_layer(input, ch_out, filter_size=1,padding=0, stride=stride, name=name)
        else:
            return input
  

def bottleneck_block( input, num_filters, stride, cardinality, name):
       
        conv0 = conv_bn_layer(input=input,num_filters=num_filters,filter_size=1,act='swish',name=name + "_branch2a")
      
        conv1_3 =conv_bn_layer(input=conv0,num_filters=num_filters,filter_size=3,
            
            padding=1,stride=stride,groups=cardinality,act='swish',name=name + "_branch2b_1")
         
        conv1_1 =conv_bn_layer(input=conv0,num_filters=num_filters,filter_size=(3,1),
            
            padding=(1,0),stride=stride,groups=cardinality,act='swish',name=name + "_branch2b_2")
        
        conv1_2 =conv_bn_layer(input=conv0,num_filters=num_filters,filter_size=(1,3),
            
            padding=(0,1),stride=stride,groups=cardinality,act='swish',name=name + "_branch2b_3")
        conv11=fluid.layers.elementwise_add(x=conv1_3, y=conv1_1,axis=1,act=None, name=name + ".add1")   
        conv1=fluid.layers.elementwise_add(x=conv11, y=conv1_2,axis=1, act=None, name=name + ".add2")
        conv2_3 =conv_bn_layer(input=conv0,num_filters=num_filters,filter_size=3,
            
            padding=1,stride=stride,dilation=2,groups=cardinality,act='swish',name=name + "_branch2a_1")
         
        conv2_1 =conv_bn_layer(input=conv0,num_filters=num_filters,filter_size=(3,1),
            
            padding=(1,0),stride=stride,dilation=2,groups=cardinality,act='swish',name=name + "_branch2a_2")
        
        conv2_2 =conv_bn_layer(input=conv0,num_filters=num_filters,filter_size=(1,3),
            
            padding=(0,1),stride=stride,dilation=2,groups=cardinality,act='swish',name=name + "_branch2a_3")
        conv22=fluid.layers.elementwise_add(x=conv2_3, y=conv2_1,axis=1,act=None, name=name + "add1")   
        conv2=fluid.layers.elementwise_add(x=conv22, y=conv2_2,axis=1, act=None, name=name + "add2")
        conv3=fluid.layers.elementwise_add(x=conv1, y=conv2,axis=1, act=None, name=name + "1add2")
        
        reduction_ratio=16
        pool = fluid.layers.pool2d(input=conv3, pool_type='avg', global_pooling=True, use_cudnn=False)
      
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
    
        squeeze = fluid.layers.fc(input=pool,size=num_filters// reduction_ratio,act='swish',
                                    param_attr=fluid.ParamAttr(
                                    initializer=fluid.initializer.Uniform(-stdv, stdv),
                                    name=name + '_sqz_weights'),
                                    bias_attr=ParamAttr(name=name + '_sqz_offset'))

        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
    
        excitation = fluid.layers.fc(input=squeeze,size=num_filters,act='sigmoid',
                                    param_attr=fluid.param_attr.ParamAttr(
                                    initializer=fluid.initializer.Uniform(-stdv, stdv),
                                    name=name + '_exc_weights'),
                                    bias_attr=ParamAttr(name=name + '_exc_offset'))
    
        sk_feature=fluid.layers.elementwise_mul(x=conv3, y=excitation, axis=0)
               
        conv2 = conv_bn_layer(input=sk_feature,num_filters=num_filters if cardinality == 64 else num_filters * 2,

                    filter_size=1,act=None,name=name + "_branch2c")
        
        short = shortcut(input,num_filters if cardinality == 64 else num_filters * 2,stride,name=name + "_branch1")
        
        return fluid.layers.elementwise_add(x=short, y=conv2, act='swish', name=name + ".add.output.5")
def net(input, class_dim=1000):

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
        
        conv = conv_bn_layer(input=input,num_filters=64,filter_size=7,padding=3,stride=2,act='swish',
                    name="res_conv1") 
      
        conv = fluid.layers.pool2d(input=conv,pool_size=3,pool_stride=2,pool_padding=1,pool_type='max')
        

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

        pool = fluid.layers.pool2d(input=conv, pool_type='avg', global_pooling=True)
        
        #drop = fluid.layers.dropout(x=pool, dropout_prob=0.5)

        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)

        out = fluid.layers.fc(input=pool,size=class_dim,act='softmax',
                                param_attr=fluid.param_attr.ParamAttr(

                                initializer=fluid.initializer.Uniform(-stdv, stdv),

                                name='fc_weights'),

                                bias_attr=fluid.param_attr.ParamAttr(name='fc_offset'))
   

        return out
