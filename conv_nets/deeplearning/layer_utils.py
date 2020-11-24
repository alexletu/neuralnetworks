from deeplearning.layers import *
from deeplearning.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def conv_relu_norm_conv_relu_norm_pool_forward(X, W1, b1, W2, b2, conv_param1, conv_param2, pool_param, gamma1, gamma2, beta1, beta2, bn_param1, bn_param2):
    out, conv_cache1 = conv_forward_fast(X, W1, b1, conv_param1)
    out, relu_cache1 = relu_forward(out)
    out, batchnorm_cache1 = spatial_batchnorm_forward(out, gamma1, beta1, bn_param1)
    
    out, conv_cache2 = conv_forward_fast(out, W2, b2, conv_param2)
    out, relu_cache2 = relu_forward(out)
    out, batchnorm_cache2 = spatial_batchnorm_forward(out, gamma2, beta2, bn_param2)
    
    out, pool_cache = max_pool_forward_fast(out, pool_param)
    cache = (conv_cache1, relu_cache1, conv_cache2, relu_cache2, pool_cache, batchnorm_cache1, batchnorm_cache2)

    return out, cache

def conv_relu_norm_conv_relu_norm_pool_backward(dout, cache):
    conv_cache1, relu_cache1, conv_cache2, relu_cache2, pool_cache, batchnorm_cache1, batchnorm_cache2 = cache
    
    upstream = max_pool_backward_fast(dout, pool_cache)
    
    upstream, dgamma2, dbeta2 = spatial_batchnorm_backward(upstream, batchnorm_cache2)
    upstream = relu_backward(upstream, relu_cache2)
    upstream, dw2, db2 = conv_backward_fast(upstream, conv_cache2)
    
    upstream, dgamma1, dbeta1 = spatial_batchnorm_backward(upstream, batchnorm_cache1)
    upstream = relu_backward(upstream, relu_cache1)
    upstream, dw1, db1 = conv_backward_fast(upstream, conv_cache1)
    
    return upstream, dw1, db1, dw2, db2, dgamma1, dgamma2, dbeta1, dbeta2

def conv_relu_conv_relu_pool_forward(X, W1, b1, W2, b2, conv_param1, conv_param2, pool_param):
    out, conv_cache1 = conv_forward_fast(X, W1, b1, conv_param1)
    out, relu_cache1 = relu_forward(out)
    out, batchnorm_cache1 = spatial_batchnorm_forward(out, gamma1, beta1, bn_param1)
    
    out, conv_cache2 = conv_forward_fast(out, W2, b2, conv_param2)
    out, relu_cache2 = relu_forward(out)
    out, batchnorm_cache2 = spatial_batchnorm_forward(out, gamma2, beta2, bn_param2)
    
    out, pool_cache = max_pool_forward_fast(out, pool_param)
    cache = (conv_cache1, relu_cache1, conv_cache2, relu_cache2, pool_cache)

    return out, cache

def conv_relu_conv_relu_pool_backward(dout, cache):
    conv_cache1, relu_cache1, conv_cache2, relu_cache2, pool_cache  = cache
    
    upstream = max_pool_backward_fast(dout, pool_cache)
    
    upstream = relu_backward(upstream, relu_cache2)
    upstream, dw2, db2 = conv_backward_fast(upstream, conv_cache2)
    
    upstream = relu_backward(upstream, relu_cache1)
    upstream, dw1, db1 = conv_backward_fast(upstream, conv_cache1)
    
    return upstream, dw1, db1, dw2, db2

def affine_norm_relu_forward_(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by normalization followed by a ReLU

    Inputs:
    - x: Input to the affine layer (N, D)
    - w, b: Weights for the affine layer (N, D), (D,)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """

    a, fc_cache = affine_forward(x, w, b)
    b, norm_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(b)
    cache = (fc_cache, norm_cache, relu_cache)
    return out, cache

def affine_norm_relu_backward_(dout, cache):
    """
    Backward pass for the affine-norm-relu convenience layer
    """
    fc_cache, norm_cache, relu_cache = cache
    
    dx_relu = relu_backward(dout, relu_cache)
    dx_norm, dgamma, dbeta = batchnorm_backward(dx_relu, norm_cache)
    dx, dw, db = affine_backward(dx_norm, fc_cache)
    return dx, dw, db, dgamma, dbeta