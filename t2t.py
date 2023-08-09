import tensorflow as tf

# Copied from tensor2tensor


def embedding_to_padding(emb):
    """Calculates the padding mask based on which embeddings are all zero.

    We have hacked symbol_modality to return all-zero embeddings for padding.

    Args:
      emb: a Tensor with shape [..., depth].

    Returns:
      a float Tensor with shape [...]. Each element is 1 if its corresponding
      embedding vector is all zero, and is 0 otherwise.
    """
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return tf.cast(tf.equal(emb_sum, 0.0), tf.float32)


def large_compatible_negative(tensor_type):
    """Large negative number as Tensor.

    This function is necessary because the standard value for epsilon
    in this module (-1e9) cannot be represented using tf.float16

    Args:
      tensor_type: a dtype to determine the type.

    Returns:
      a large negative number.
    """
    if tensor_type == tf.float16:
        return tf.float16.min
    return -1e9


def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.

    Args:
      memory_padding: a float `Tensor` with shape [batch, memory_length].

    Returns:
      a `Tensor` with shape [batch, 1, 1, memory_length].
    """
    ret = memory_padding * large_compatible_negative(memory_padding.dtype)
    return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)


def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
    """Matrix band part of ones.

    Args:
      rows: int determining number of rows in output
      cols: int
      num_lower: int, maximum distance backward. Negative values indicate
        unlimited.
      num_upper: int, maximum distance forward. Negative values indicate
        unlimited.
      out_shape: shape to reshape output by.

    Returns:
      Tensor of size rows * cols reshaped into shape out_shape.
    """
    if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
        # Needed info is constant, so we construct in numpy
        if num_lower < 0:
            num_lower = rows - 1
        if num_upper < 0:
            num_upper = cols - 1
        lower_mask = np.tri(cols, rows, num_lower).T
        upper_mask = np.tri(rows, cols, num_upper)
        band = np.ones((rows, cols)) * lower_mask * upper_mask
        if out_shape:
            band = band.reshape(out_shape)
        band = tf.constant(band, tf.float32)
    else:
        band = tf.linalg.band_part(
            tf.ones([rows, cols]),
            tf.cast(num_lower, tf.int64),
            tf.cast(num_upper, tf.int64),
        )
        if out_shape:
            band = tf.reshape(band, out_shape)

    return band


def attention_bias_local(length, max_backward, max_forward):
    """Create an bias tensor to be added to attention logits.

    A position may attend to positions at most max_distance from it,
    forward and backwards.

    This does not actually save any computation.

    Args:
      length: int
      max_backward: int, maximum distance backward to attend. Negative values
        indicate unlimited.
      max_forward: int, maximum distance forward to attend. Negative values
        indicate unlimited.

    Returns:
      a `Tensor` with shape [1, 1, length, length].
    """
    band = ones_matrix_band_part(
        length, length, max_backward, max_forward, out_shape=[1, 1, length, length]
    )
    return -1e9 * (1.0 - band)


def attention_bias_lower_triangle(length):
    """Create an bias tensor to be added to attention logits.

    Allows a query to attend to all positions up to and including its own.

    Args:
     length: a Scalar.

    Returns:
      a `Tensor` with shape [1, 1, length, length].
    """
    return attention_bias_local(length, -1, 0)


def _scaled_dot_product_attention_simple(q, k, v, bias):
    scalar = tf.math.rsqrt(tf.cast(tf.shape(q)[2], tf.float32))
    logits = tf.matmul(q * scalar, k, transpose_b=True)
    if bias is not None:
        logits += bias
    weights = tf.nn.softmax(logits)
    return tf.matmul(weights, v)
