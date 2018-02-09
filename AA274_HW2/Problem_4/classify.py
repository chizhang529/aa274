import argparse
import os
import numpy as np
import tensorflow as tf
from utils import *

GRAPH = 'retrained_graph.pb'
with open('labels.txt', 'r') as f:
    LABELS = f.read().split()

def get_images_by_class(directory):
    image_list = []
    for label in LABELS:
        image_list.append([directory + label + '/' + fname for fname in os.listdir(directory + label) if fname.endswith('jpg')])
    return image_list

def classify(testset):
    results = []
    with tf.Session() as sess:
        for ground_truth, image_list in enumerate(testset):
            for image in image_list:
                probs = classify_image(image, sess)
                prediction = np.argmax(probs)
                results.append(prediction == ground_truth)
                if prediction != ground_truth:
                    print image, 'incorrectly classifed as', LABELS[prediction]
                    print 'P(', LABELS, ') =', probs
    accuracy = float(sum(results)) / len(results)
    print 'Test accuracy: %.1f%% (%d / %d)' % (accuracy * 100, sum(results), len(results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_image_dir', type=str, default='datasets/test/')
    FLAGS, _ = parser.parse_known_args()
    load_graph(GRAPH)
    testset = get_images_by_class(FLAGS.test_image_dir)
    classify(testset)