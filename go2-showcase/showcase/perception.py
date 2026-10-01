"""Decode raw ROS camera pixels. Semantic perception belongs to the vision model."""

def decode_image(message):
    """Respect ROS Image row strides and endianness; fail on unknown encodings."""
    import numpy as np
    kinds = {'rgb8': ('u1', 3), 'bgr8': ('u1', 3), '32FC1': ('f4', 1), '16UC1': ('u2', 1)}
    if message.encoding not in kinds:
        raise ValueError('Unsupported camera encoding '+message.encoding)
    dtype, channels = kinds[message.encoding]
    dtype = np.dtype(('>' if message.is_bigendian else '<')+dtype)
    width_bytes = message.width * channels * dtype.itemsize
    if message.step < width_bytes or len(message.data) < message.step * message.height:
        raise ValueError('Truncated camera image')
    rows = np.frombuffer(bytes(message.data), dtype='u1').reshape(message.height, message.step)
    image = np.ascontiguousarray(rows[:, :width_bytes]).view(dtype)
    image = image.reshape(message.height, message.width, channels) if channels > 1 else image.reshape(message.height, message.width)
    if message.encoding == 'bgr8':
        image = image[:, :, ::-1].copy()
    if message.encoding == '16UC1':
        image = image.astype('f4') / 1000
    return image
