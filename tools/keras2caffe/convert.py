#coding=utf-8
import caffe
from caffe import layers as L, params as P
import math
import numpy as np


def set_padding(config_keras, input_shape, config_caffe):
    if config_keras['padding']=='valid':
        return
    elif config_keras['padding']=='same':
        
        if 'kernel_size' in config_keras:
            kernel_size = config_keras['kernel_size']
        elif 'pool_size' in config_keras:
            kernel_size = config_keras['pool_size']
        else:
            raise Exception('Undefined kernel size')
        
        strides = config_keras['strides']
        w = input_shape[1]
        h = input_shape[2]
        
        out_w = math.ceil(w / float(strides[1]))
        pad_w = int((kernel_size[1]*out_w - (kernel_size[1]-strides[1])*(out_w - 1) - w)/2)
        
        out_h = math.ceil(h / float(strides[0]))
        pad_h = int((kernel_size[0]*out_h - (kernel_size[0]-strides[0])*(out_h - 1) - h)/2)
        
        if pad_w==0 and pad_h==0:
            return
        
        if pad_w==pad_h:
            config_caffe['pad'] = pad_w
        else:
            config_caffe['pad_h'] = pad_h
            config_caffe['pad_w'] = pad_w
        
    else:
        raise Exception(config_keras['padding']+' padding is not supported')

def convert(keras_model, caffe_net_file, caffe_params_file):
    
    caffe_net = caffe.NetSpec()
    
    net_params = dict()
    
    outputs=dict()
    shape=()
    
    input_str = ''
    
    for layer in keras_model.layers:
        name = layer.name
        layer_type = type(layer).__name__
        
        config = layer.get_config()

        blobs = layer.get_weights()
        blobs_num = len(blobs)
        
        if type(layer.output)==list:
            raise Exception('Layers with multiply outputs are not supported')
        else: 
            top=layer.output.name
        
        if type(layer.input)!=list:
            bottom = layer.input.name
        
        if layer_type=='InputLayer' or len(caffe_net.tops)==0:

            input_name = 'data'
            caffe_net[input_name] = L.Layer()
            input_shape = config['batch_input_shape']
            input_str = 'input: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}'.format('"' + input_name + '"',
                1, input_shape[3], input_shape[1], input_shape[2])
            outputs[layer.input.name] = input_name
            if layer_type=='InputLayer':
                continue
                
        if layer_type=='Conv2D' or layer_type=='Convolution2D':
            
            strides = config['strides']
            kernel_size = config['kernel_size']
            
            kwargs = { 'num_output': config['filters'] }
            
            if kernel_size[0]==kernel_size[1]:
            	kwargs['kernel_size']=kernel_size[0]
            else:
            	kwargs['kernel_h']=kernel_size[0]
            	kwargs['kernel_w']=kernel_size[1]
            
            if strides[0]==strides[1]:
            	kwargs['stride']=strides[0]
            else:
            	kwargs['stride_h']=strides[0]
            	kwargs['stride_w']=strides[1]
            
            if not config['use_bias']:
            	kwargs['bias_term'] = False
            	#kwargs['param']=[dict(lr_mult=0)]
            else:
                #kwargs['param']=[dict(lr_mult=0), dict(lr_mult=0)]
                pass
            
            set_padding(config, layer.input_shape, kwargs)
            
            caffe_net[name] = L.Convolution(caffe_net[outputs[bottom]], **kwargs)
            
            blobs[0] = np.array(blobs[0]).transpose(3,2,0,1)
            net_params[name] = blobs

            if config['activation'] == 'relu':
                name_s = name+'s'
                caffe_net[name_s] = L.ReLU(caffe_net[name], in_place=True)
            elif config['activation'] == 'sigmoid':
                name_s = name+'s'
                caffe_net[name_s] = L.Sigmoid(caffe_net[name], in_place=True)
            elif config['activation'] == 'linear':
                #do nothing
                pass
            else:
                raise Exception('Unsupported activation '+config['activation'])
        
        elif layer_type=='Conv2DTranspose':

            stride = config['strides']
            kernel_size = config['kernel_size']
            channels = config['filters']
            group = config['group']

            w = layer.input_shape[1]
            h = layer.input_shape[2]
        
            out_w = math.ceil(w / float(stride[1]))
            pad_w = int((kernel_size[1]*out_w - (kernel_size[1]-strides[1])*(out_w - 1) - w)/2)
        
            out_h = math.ceil(h / float(strides[0]))
            pad_h = int((kernel_size[0]*out_h - (kernel_size[0]-strides[0])*(out_h - 1) - h)/2)

            if not config['use_bias']:
                bias_flag = False
            else:
                bias_flag = True

            if pad_w == 0:
                
                caffe_net[name] = L.Deconvolution(caffe_net[outputs[bottom]], convolution_param=dict(num_output=channels, 
                group=channels, kernel_size=kernel_size, stride=stride, weight_filler=dict(type='bilinear'), 
                bias_term=bias_flag), param=dict(lr_mult=0, decay_mult=0))

            else:

                if pad_w==pad_h:
                    config_caffe['pad'] = pad_w
                else:
                    config_caffe['pad_h'] = pad_h
                    config_caffe['pad_w'] = pad_w
            
                caffe_net[name] = L.Deconvolution(caffe_net[outputs[bottom]], convolution_param=dict(num_output=channels, 
                group=channels, kernel_size=kernel_size, stride=stride, pad=pad, weight_filler=dict(type='bilinear'), 
                bias_term=bias_flag), param=dict(lr_mult=0, decay_mult=0))
            
            blob = np.array(blobs[0]).transpose(2, 3, 0, 1)
            blob.shape = (1,) + blob.shape
            net_params[name] = blob

            if config['activation'] == 'relu':
                name_s = name+'s'
                caffe_net[name_s] = L.ReLU(caffe_net[name], in_place=True)
            elif config['activation'] == 'sigmoid':
                name_s = name+'s'
                caffe_net[name_s] = L.Sigmoid(caffe_net[name], in_place=True)
            elif config['activation'] == 'linear':
                #do nothing
                pass
            else:
                raise Exception('Unsupported activation '+config['activation'])

        # 深度可分离卷积
        elif layer_type=='DepthwiseConv2D':
            
            strides = config['strides']
            kernel_size = config['kernel_size']

            kwargs = {'num_output': layer.input_shape[3]}

            if kernel_size[0] == kernel_size[1]:
                kwargs['kernel_size'] = kernel_size[0]
            else:
                kwargs['kernel_h'] = kernel_size[0]
                kwargs['kernel_w'] = kernel_size[1]

            if strides[0] == strides[1]:
                kwargs['stride'] = strides[0]
            else:
                kwargs['stride_h'] = strides[0]
                kwargs['stride_w'] = strides[1]

            set_padding(config, layer.input_shape, kwargs)

            kwargs['group'] = layer.input_shape[3]

            kwargs['bias_term'] = False
            caffe_net[name] = L.Convolution(caffe_net[outputs[bottom]], **kwargs)
            blob = np.array(blobs[0]).transpose(2, 3, 0, 1)
            blob.shape = (1,) + blob.shape
            net_params[name] = blob
            
            if config['activation'] == 'relu':
                name_s = name+'s'
                caffe_net[name_s] = L.ReLU(caffe_net[name], in_place=True)
            elif config['activation'] == 'sigmoid':
                name_s = name+'s'
                caffe_net[name_s] = L.Sigmoid(caffe_net[name], in_place=True)
            elif config['activation'] == 'linear':
                #do nothing
                pass
            else:
                raise Exception('Unsupported activation '+config['activation'])

        elif layer_type == 'SeparableConv2D':

            strides = config['strides']
            kernel_size = config['kernel_size']

            kwargs = {'num_output': layer.input_shape[3]}

            if kernel_size[0] == kernel_size[1]:
                kwargs['kernel_size'] = kernel_size[0]
            else:
                kwargs['kernel_h'] = kernel_size[0]
                kwargs['kernel_w'] = kernel_size[1]

            if strides[0] == strides[1]:
                kwargs['stride'] = strides[0]
            else:
                kwargs['stride_h'] = strides[0]
                kwargs['stride_w'] = strides[1]

            set_padding(config, layer.input_shape, kwargs)

            kwargs['group'] = layer.input_shape[3]

            kwargs['bias_term'] = False
            caffe_net[name] = L.Convolution(caffe_net[outputs[bottom]], **kwargs)
            blob = np.array(blobs[0]).transpose(2, 3, 0, 1)
            blob.shape = (1,) + blob.shape
            net_params[name] = blob

            name2 = name + '_'
            kwargs = {'num_output': config['filters'], 'kernel_size': 1, 'bias_term': config['use_bias']}
            caffe_net[name2] = L.Convolution(caffe_net[name], **kwargs)

            if config['use_bias'] == True:
                blob2 = []
                blob2.append(np.array(blobs[1]).transpose(3, 2, 0, 1))
                blob2.append(np.array(blobs[2]))
                blob2[0].shape = (1,) + blob2[0].shape
            else:
                blob2 = np.array(blobs[1]).transpose(3, 2, 0, 1)
                blob2.shape = (1,) + blob2.shape

            net_params[name2] = blob2
            name = name2

        elif layer_type=='BatchNormalization':
            
            param = dict()
            
            variance = np.array(blobs[-1])
            mean = np.array(blobs[-2])
            
            if config['scale']:
                gamma = np.array(blobs[0])
                sparam=[dict(lr_mult=1), dict(lr_mult=1)]
            else:
                gamma = np.ones(mean.shape, dtype=np.float32)
                #sparam=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=1, decay_mult=1)]
                sparam=[dict(lr_mult=0), dict(lr_mult=1)]
                #sparam=[dict(lr_mult=0), dict(lr_mult=0)]
            
            if config['center']:
                beta = np.array(blobs[-3])
                param['bias_term']=True
            else:
                beta = np.zeros(mean.shape, dtype=np.float32)
                param['bias_term']=False
            
            caffe_net[name] = L.BatchNorm(caffe_net[outputs[bottom]], in_place=True)
            	#param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
            	#param=[dict(lr_mult=1), dict(lr_mult=1), dict(lr_mult=0)])
                
            net_params[name] = (mean, variance, np.array(1.0)) 
            
            name_s = name+'s'
            
            caffe_net[name_s] = L.Scale(caffe_net[name], in_place=True, 
            	param=sparam, scale_param={'bias_term': config['center']})
            net_params[name_s] = (gamma, beta)
            
        elif layer_type=='Dense':
            caffe_net[name] = L.InnerProduct(caffe_net[outputs[bottom]], 
            	num_output=config['units'], weight_filler=dict(type='xavier'))
            
            if config['use_bias']:
                weight=np.array(blobs[0]).transpose(1, 0)
                if type(layer._inbound_nodes[0].inbound_layers[0]).__name__=='Flatten':
                    flatten_shape=layer._inbound_nodes[0].inbound_layers[0].input_shape
                    for i in range(weight.shape[0]):
                        weight[i]=np.array(weight[i].reshape(flatten_shape[1],flatten_shape[2],flatten_shape[3]).transpose(2,0,1).reshape(weight.shape[1]))
                net_params[name] = (weight, np.array(blobs[1]))
            else:
                net_params[name] = (blobs[0])
                
            name_s = name+'s'
            if config['activation']=='softmax':
                caffe_net[name_s] = L.Softmax(caffe_net[name], in_place=True)
            elif config['activation']=='relu':
                caffe_net[name_s] = L.ReLU(caffe_net[name], in_place=True)
        
        elif layer_type=='Activation':
            if config['activation']=='relu':
                #caffe_net[name] = L.ReLU(caffe_net[outputs[bottom]], in_place=True)
                if len(layer.input.consumers())>1:
                    caffe_net[name] = L.ReLU(caffe_net[outputs[bottom]])
                else:
                    caffe_net[name] = L.ReLU(caffe_net[outputs[bottom]], in_place=True)
            elif config['activation']=='relu6':
                #TODO
                caffe_net[name] = L.ReLU(caffe_net[outputs[bottom]])
            elif config['activation']=='softmax':
                caffe_net[name] = L.Softmax(caffe_net[outputs[bottom]], in_place=True)
            elif config['activation'] == 'sigmoid':
                # name_s = name+'s'
                caffe_net[name] = L.Sigmoid(caffe_net[outputs[bottom]], in_place=True)
            else:
                raise Exception('Unsupported activation '+config['activation'])
        
        elif layer_type=='Cropping2D':
            shape = layer.output_shape
            ddata = L.DummyData(shape=dict(dim=[1, shape[3],shape[1], shape[2]]))
            layers = []
            layers.append(caffe_net[outputs[bottom]])   
            layers.append(ddata)   #TODO
            caffe_net[name] = L.Crop(*layers)
        
        elif layer_type=='Concatenate' or layer_type=='Merge':
            layers = []
            for i in layer.input:
                layers.append(caffe_net[outputs[i.name]])
            caffe_net[name] = L.Concat(*layers, axis=1)
        
        elif layer_type=='Add':
            layers = []
            for i in layer.input:
                layers.append(caffe_net[outputs[i.name]])
            caffe_net[name] = L.Eltwise(*layers)
        
        elif layer_type=='Flatten':
            caffe_net[name] = L.Flatten(caffe_net[outputs[bottom]])
        
        elif layer_type=='Reshape':
            shape = config['target_shape']
            if len(shape)==3:
                #shape = (layer.input_shape[0], shape[2], shape[0], shape[1])
                shape = (1, shape[2], shape[0], shape[1])
            elif len(shape)==1:
                #shape = (layer.input_shape[0], 1, 1, shape[0])
                shape = (1, 1, 1, shape[0])
            elif len(shape)==2:
                shape = (0, shape[1], -1, 0)
            caffe_net[name] = L.Reshape(caffe_net[outputs[bottom]], 
                reshape_param={'shape':{'dim': list(shape)}})
        
        elif layer_type=='MaxPooling2D' or layer_type=='AveragePooling2D':
            
            kwargs={}
            
            if layer_type=='MaxPooling2D':
                kwargs['pool'] = P.Pooling.MAX
            else:
                kwargs['pool'] = P.Pooling.AVE
                
            pool_size = config['pool_size']
            strides  = config['strides']
            
            if pool_size[0]!=pool_size[1]:
                raise Exception('Unsupported pool_size')
                    
            if strides[0]!=strides[1]:
                raise Exception('Unsupported strides')
            
            set_padding(config, layer.input_shape, kwargs)
            
            caffe_net[name] = L.Pooling(caffe_net[outputs[bottom]], kernel_size=pool_size[0], 
                stride=strides[0], **kwargs)
        

        elif layer_type=='Dropout':
            caffe_net[name] = L.Dropout(caffe_net[outputs[bottom]], 
                dropout_param=dict(dropout_ratio=config['rate']))
        
        elif layer_type=='GlobalAveragePooling2D':
            caffe_net[name] = L.Pooling(caffe_net[outputs[bottom]], pool=P.Pooling.AVE, 
                pooling_param=dict(global_pooling=True))
        
        elif layer_type=='UpSampling2D':
            if config['size'][0]!=config['size'][1]:
                raise Exception('Unsupported upsampling factor')
            factor = config['size'][0]
            kernel_size = 2 * factor - factor % 2
            stride = factor
            pad = int(math.ceil((factor - 1) / 2.0))
            channels = layer.input_shape[-1]
            caffe_net[name] = L.Deconvolution(caffe_net[outputs[bottom]], convolution_param=dict(num_output=channels, 
                group=channels, kernel_size=kernel_size, stride=stride, pad=pad, weight_filler=dict(type='bilinear'), 
                bias_term=False), param=dict(lr_mult=0, decay_mult=0))
        
        elif layer_type=='LeakyReLU':
            caffe_net[name] = L.ReLU(caffe_net[outputs[bottom]], negative_slope=config['alpha'], in_place=True)

        # Caffe中没有ZeroPadding2D存在，因此需要避免这个Op的应用，即将Padding写进卷积/反卷积/Pooling等层中
        #elif layer_type=='ZeroPadding2D':
            # padding=config['padding']
            # caffe_net[name] = L.Pooling(caffe_net[outputs[bottom]], kernel_size=1, 
            #     stride=1, pad_h=padding[0][0]+padding[0][1], pad_w=padding[1][0]+padding[1][1], pool=P.Pooling.AVE)
        
        else:
            raise Exception('Unsupported layer type: '+layer_type)
            
        outputs[top]=name
        
    
    #replace empty layer with input blob
    net_proto = input_str + '\n' + 'layer {' + 'layer {'.join(str(caffe_net.to_proto()).split('layer {')[2:])
    
    f = open(caffe_net_file, 'w') 
    f.write(net_proto)
    f.close()
    
    caffe_model = caffe.Net(caffe_net_file, caffe.TEST)
    
    for layer in caffe_model.params.keys():
        if 'up_sampling2d' in layer:
            continue
        for n in range(0, len(caffe_model.params[layer])):
            caffe_model.params[layer][n].data[...] = net_params[layer][n]

    caffe_model.save(caffe_params_file)
