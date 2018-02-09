import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from retrain import MODEL_TYPE, add_jpeg_decoding, create_model_info

MODEL_INFO = create_model_info(MODEL_TYPE)

def load_graph(model_path):
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    add_jpeg_decoding(MODEL_INFO['input_width'], MODEL_INFO['input_height'],
                      MODEL_INFO['input_depth'], MODEL_INFO['input_mean'],
                      MODEL_INFO['input_std'])
    graph = tf.get_default_graph()
    if MODEL_TYPE.startswith('mobilenet'):
        final_fc_weights = graph.get_tensor_by_name('MobilenetV1/Logits/Conv2d_1c_1x1/weights:0')          # (1, 1, 512, 1001)
        final_fc_biases = graph.get_tensor_by_name('MobilenetV1/Logits/Conv2d_1c_1x1/biases:0')            # (1001)
        conv_7x7_output = graph.get_tensor_by_name('MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:0')  # (1, 7, 7, 512)
        bottleneck_list = tf.matmul(tf.reshape(conv_7x7_output, [-1, conv_7x7_output.shape[-1]]),
                                    tf.squeeze(final_fc_weights)) + final_fc_biases
        tf.reshape(bottleneck_list, [7, 7, -1], name='bottleneck_grid')    # (7, 7, 1001)
    elif MODEL_TYPE == 'inception_v3':
        tf.reshape(graph.get_tensor_by_name('mixed_10/join:0'), [8, 8, -1], name='bottleneck_grid')    # (8, 8, 2048)
    else:
        raise ValueError('Unknown MODEL_TYPE: ' + MODEL_TYPE)

def input_tensor_and_image_data(image):
    graph = tf.get_default_graph()
    if isinstance(image, str):
        return graph.get_tensor_by_name('DecodeJPGInput:0'), tf.gfile.FastGFile(image, 'rb').read()
    else:
        return graph.get_tensor_by_name('JPG_uint8:0'), image

def run_with_image_input(tensor, image, sess=None):
    # CNN_input and normalized_image hold the same values, but are not actually the same tensor in the TF graph
    # so we have to do these gymnastics instead of just calling sess.run(tensor, {input_tensor: raw_image_data}) once
    graph = tf.get_default_graph()
    normalized_image = graph.get_tensor_by_name('NormalizedImage:0')
    CNN_input = graph.get_tensor_by_name(MODEL_INFO['resized_input_tensor_name'])
    input_tensor, raw_image_data = input_tensor_and_image_data(image)
    if sess:
        image_data = sess.run(normalized_image, {input_tensor: raw_image_data})
        return sess.run(tensor, {CNN_input: image_data, input_tensor: raw_image_data})
    else:
        with tf.Session() as sess:
            image_data = sess.run(normalized_image, {input_tensor: raw_image_data})
            return sess.run(tensor, {CNN_input: image_data, input_tensor: raw_image_data})

def decode_jpeg(image_path, sess=None):
    # a bit overkill to use tensorflow for this, but we'll use its jpeg decoder to ensure consistency;
    # likely equivalent to matplotlib.image.imread
    return run_with_image_input(tf.get_default_graph().get_tensor_by_name('JPG_uint8:0'), image_path, sess)

def classify_image(image, sess=None):
    class_probs = run_with_image_input(tf.get_default_graph().get_tensor_by_name('final_result:0'), image, sess)
    class_probs = np.squeeze(class_probs)
    return class_probs

def gradient_of_class_score_with_respect_to_input_image(image, class_, sess):
    graph = tf.get_default_graph()
    logits_tensor = graph.get_tensor_by_name('final_training_ops/Wx_plus_b/logits:0')
    CNN_input = graph.get_tensor_by_name(MODEL_INFO['resized_input_tensor_name'])
    normalized_image = graph.get_tensor_by_name('NormalizedImage:0')
    unnormalized_image = graph.get_tensor_by_name('JPG_float32:0')

    # We have to do a bit of manual gradient propagation to get the gradients with respect to the original input image
    # CNN_input and normalized_image hold the same values, but are not actually the same tensor in the TF graph
    gradient_tensor = tf.gradients(logits_tensor[0,class_], CNN_input)
    gradients_wrt_CNN_input = run_with_image_input(gradient_tensor, image, sess)
    gradient_tensor = tf.gradients(normalized_image, unnormalized_image, gradients_wrt_CNN_input)
    gradients_wrt_raw_input_image = run_with_image_input(gradient_tensor, image, sess)
    w_ijc = np.squeeze(gradients_wrt_raw_input_image)
    return w_ijc