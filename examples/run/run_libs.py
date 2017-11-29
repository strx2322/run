import caffe
from caffe import layers as L
from caffe.model_libs import UnpackVariable, ConvBNLayer

def DeconvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    param = [dict(lr_mult=lr_mult, decay_mult=1)]
    eps = bn_params.get('eps', 0.001)
    moving_average_fraction = bn_params.get('moving_average_fraction', 0.999)
    use_global_stats = bn_params.get('use_global_stats', False)
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        'moving_average_fraction': moving_average_fraction,
        }
    bn_lr_mult = lr_mult
    if use_global_stats:
      # only specify if use_global_stats is explicitly provided;
      # otherwise, use_global_stats_ = this->phase_ == TEST;
      bn_kwargs = {
          'param': [
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0)],
          'eps': eps,
          'use_global_stats': use_global_stats,
          }
      # not updating scale/bias parameters
      bn_lr_mult = 0
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [
              dict(lr_mult=bn_lr_mult, decay_mult=0),
              dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }
    param = [
            dict(lr_mult=lr_mult, decay_mult=1),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    convolution_param=dict(num_output=num_output, kernel_size=kernel_h, pad=pad_h, 
                                    stride=stride_h)
  else:
    convolution_param=dict(num_output=num_output,
                                    kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                                    stride_h=stride_h, stride_w=stride_w)
  convolution_param.update(kwargs)
  net[conv_name] = L.Deconvolution(net[from_layer],
                                    convolution_param=convolution_param, param=param)

  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)


def Res3Way(net, from_layer, deconv_layer, block_name, use_branch, freeze_branch=[False,False,False], use_bn=True, *branch_param):
    
    res_layer = []
    if use_branch[0]:
        branch1 = ResBranch(net, from_layer, block_name, "branch1", freeze_branch[0], branch_param[0], use_bn=use_bn)
        res_layer.append(net[branch1])
    else:
        res_layer.append(net[from_layer])

    if use_branch[1]:
        branch2 = ResBranch(net, from_layer, block_name, "branch2", freeze_branch[1], branch_param[1], use_bn=use_bn)
        res_layer.append(net[branch2])

    if use_branch[2]:
        branch3 = ResBranch(net, deconv_layer, block_name, "branch3", freeze_branch[2], branch_param[2], use_bn=use_bn)
        res_layer.append(net[branch3])

    res_name = 'res{}'.format(block_name)

    if len(res_layer) != 1:
        net[res_name] = L.Eltwise(*res_layer)
        relu_name = '{}_relu'.format(res_name)
        net[relu_name] = L.ReLU(net[res_name], in_place=True)
    else:
        relu_name = '{}_relu'.format(res_name)
        net[relu_name] = L.ReLU(res_layer[0], in_place=True)

    return relu_name

def ResBranch(net, from_layer, block_name, branch_prefix, freeze, layer_param, use_bn=True, **bn_params):
    conv_prefix = 'res{}_'.format(block_name)
    conv_postfix = ''
    bn_prefix = 'bn{}_'.format(block_name)
    bn_postfix = ''
    scale_prefix = 'scale{}_'.format(block_name)
    scale_postfix = ''
    use_scale = True

    if freeze:
        lr_mult = 0
        decay_mult = 0
    else:
        lr_mult = 1
        decay_mult = 1

    num_layers = len(layer_param)

    if num_layers != 1:
        name_postfix = ['a', 'b', 'c', 'd', 'e']
    else:
        name_postfix = ['']
    id = 0
    out_name = from_layer

    for param in layer_param:

        branch_name = branch_prefix + name_postfix[id]
        id += 1

        num_output = param['out']
        kernel_size = param['kernel_size']
        pad = param['pad']
        stride = param['stride']
        use_relu = id is not num_layers

        if param['name'] == 'Convolution':
            ConvBNLayer(net, out_name, branch_name, use_bn=use_bn, use_relu=use_relu,
                num_output=num_output, kernel_size=kernel_size, pad=pad, stride=stride, use_scale=use_scale,
                lr_mult=lr_mult,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_params)
        elif param['name'] == 'Deconvolution':
            DeconvBNLayer(net, out_name, branch_name, use_bn=use_bn, use_relu=use_relu,
                num_output=num_output, kernel_size=kernel_size, pad=pad, stride=stride, use_scale=use_scale,
                lr_mult=lr_mult,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_params)

        out_name = '{}{}'.format(conv_prefix, branch_name)

    return out_name


def CreateUnifiedPredictionHead(net, data_layer="data", num_classes=[], from_layers=[],
                                use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
                                use_scale=True, min_sizes=[], max_sizes=[], prior_variance=[0.1],
                                aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
                                flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
                                conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []

    loc_args = {
        'param': [
            dict(name='loc_p1', lr_mult=lr_mult, decay_mult=1),
            dict(name='loc_p2', lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
    }

    conf_args = {
        'param': [
            dict(name='conf_p1', lr_mult=lr_mult, decay_mult=1),
            dict(name='conf_p2', lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
    }

    if flip:
        num_priors_per_location = 6
    else:
        num_priors_per_location = 3

    for i in range(0, num):
        from_layer = from_layers[i]

        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)

        # Create location prediction layer.
        net[name] = L.Convolution(net[from_layer], num_output=num_priors_per_location * 4,
                                  pad=1, kernel_size=3, stride=1, **loc_args)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        net[name] = L.Convolution(net[from_layer], num_output=num_priors_per_location * num_classes,
                                  pad=1, kernel_size=3, stride=1, **conf_args)

        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                               clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                        num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers

def ShareMboxParam(net, from_layer):
    for layer in from_layer:
        cls_layer_name = "{}_mbox_conf".format(layer)
        loc_layer_name = "{}_mbox_loc".format(layer)

        cls_param = net[cls_layer_name].fn.params["param"]
        loc_param = net[loc_layer_name].fn.params["param"]

        cls_param[0].update({'name': 'shared_mbox_conf_filters'})
        cls_param[1].update({'name': 'shared_mbox_conf_biases'})
        loc_param[0].update({'name': 'shared_mbox_loc_filters'})
        loc_param[1].update({'name': 'shared_mbox_loc_biases'})


