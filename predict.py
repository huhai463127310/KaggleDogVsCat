# -*- coding: utf-8 -*-

"""
预测
"""

import time
import os
import csv
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from lib import inception_resnet_v2

from pre_process import read_test_image
from pre_process import config_log

def predict(output_file, model, image_dir, checkpoint_file_or_dir, log_dir = "test_logs/", test_image_num = 12500):
    image_path = os.path.join(image_dir, "*.jpg")    
    
    BATCH_SIZE = 32
    BATCH_THREADS = 12
    BATCH_CAPACITY = BATCH_SIZE*2

    step = 0
    max_step = math.ceil(test_image_num/BATCH_SIZE)
    count = 0
    
    config_log(log_dir + "/" + model + ".log")

    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True

    with open(output_file, "w", buffering=4096) as fp:
        writer = csv.writer(fp) 
        writer.writerow(["id","label"])
        with tf.Graph().as_default():
            with tf.Session(config=config) as sess:
                
                if model == "vgg_16":
                    image_h, image_w = 224, 224
                    model_fn = nets.vgg.vgg_16
                    inputs, files = read_test_image(image_path, image_h, image_w, epochs=1, batch_size=BATCH_SIZE, 
                                    batch_threads=BATCH_THREADS, batch_capacity=BATCH_CAPACITY)
                    predictions, _ = model_fn(inputs, num_classes=2, is_training=False)
                elif model == "inception_resnet_v2":
                    image_h, image_w = 299, 299                   
                    model_fn = inception_resnet_v2.inception_resnet_v2
                    inputs, files = read_test_image(image_path, image_h, image_w, epochs=1, batch_size=BATCH_SIZE, 
                                    batch_threads=BATCH_THREADS, batch_capacity=BATCH_CAPACITY)
                    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
                        predictions, _ = model_fn(inputs, num_classes=2, is_training=False)
                else:
                    raise ValueError("model {} not supported".format(model))               
                    
                variables_to_restore = slim.get_model_variables()
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

                restorer = tf.train.Saver(variables_to_restore)
                
                if os.path.isdir(checkpoint_file_or_dir):
                    ckpt = tf.train.latest_checkpoint(checkpoint_file_or_dir)
                else:
                    ckpt = checkpoint_file_or_dir
                
                tf.logging.info("found ckpt : {}".format(ckpt))
                restorer.restore(sess, ckpt)


                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                try:
                    while not coord.should_stop():
                        if count >= test_image_num:
                            break
                        ret = tf.nn.softmax(predictions)
                        ret = tf.clip_by_value(ret, 0.005, 0.995)
                        preds, img_ids = sess.run([ret, files])
                        tf.logging.info("step = {}".format(step))
                        for i, pred in enumerate(preds):
                            writer.writerow([int(img_ids[i]), pred[1]])

                        count += len(img_ids)
                        step += 1
                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                finally:
                    coord.request_stop()

                coord.join(threads)

if __name__ == "__main__":
    local_time = time.localtime(time.time())
    time_str = time.strftime("%Y%m%d%H%M", local_time)
    flags = tf.app.flags
    flags.DEFINE_string("output_file", "output/predict_test_" + time_str + ".csv", "output file path")
    flags.DEFINE_string("model", "", "model:vgg_16, inception_resnet_v2")
    flags.DEFINE_string("image_dir", "input/test", "image directory")
    flags.DEFINE_string("checkpoint", "", "checkpoint directory or file")
    args = flags.FLAGS
    
    if args.model not in ["vgg_16", "inception_resnet_v2"]:
        print("model {} not supported".format(args.model))
        exit(1)
    if not os.path.exists(args.image_dir):
        print("image_dir {} not exits".format(args.image_dir))
        exit(1)
    predict(args.output_file, args.model, args.image_dir, args.checkpoint)