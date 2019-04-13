# -*- coding: utf-8 -*-

"""
预处理
"""

import logging
import tensorflow as tf

def config_log(file_name):
    logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s %(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=file_name,
                    filemode='w')


def read_image(image_dir_path, image_height, image_weight, epochs=1, batch_size=2, batch_threads=32, batch_capacity = 1000, data_augmentation=False):
    """
    读取数据
    :return: (图片批次op,标签批次op,图片名称)
    """
    file_names = tf.train.match_filenames_once(image_dir_path)    
    filename_queue = tf.train.string_input_producer(file_names, num_epochs=epochs, shuffle=False)
    image_reader = tf.WholeFileReader()
    key, image_file = image_reader.read(filename_queue)

    label_name = tf.string_split([key], delimiter=".").values[0]
    label_name = tf.string_split([label_name], delimiter="/").values[-1]
    # 根据名字是dog还是cat分配标签，dog:1,cat:0
    label = tf.cond(tf.equal(label_name, "dog"), lambda: [0., 1.] , lambda: [1., 0.])

    image = tf.image.decode_jpeg(image_file, channels=3)
    
    if data_augmentation:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        
    image = tf.image.convert_image_dtype(image, tf.float32)
#     #图像大小调整
    image = tf.image.resize_image_with_crop_or_pad(image, image_height, image_weight)

    # 数据分批
    image_batch, label_batch, name_batch = tf.train.batch([image, label, key], batch_size=batch_size, num_threads=batch_threads, capacity=batch_capacity, allow_smaller_final_batch=True)
    
    return image_batch, label_batch, name_batch

def read_test_image(image_dir_path, image_height, image_weight, epochs=1, batch_size=2, batch_threads=32, 
               batch_capacity = 1000):
    """
    读取测试数据
    :return: (图片批次op,标签批次op)
    """
    file_names = tf.train.match_filenames_once(image_dir_path)    
    filename_queue = tf.train.string_input_producer(file_names, num_epochs=epochs, shuffle=False)
    image_reader = tf.WholeFileReader()
    key, image_file = image_reader.read(filename_queue)

    label_name = tf.string_split([key], delimiter=".").values[0]
    label_name = tf.string_split([label_name], delimiter="/").values[-1]
    # 根据名字是dog还是cat分配标签，dog:1,cat:0
    file_id = label_name


    image = tf.image.decode_jpeg(image_file, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # 图像大小调整
    image = tf.image.resize_image_with_crop_or_pad(image, image_height, image_weight)

    # 数据分批   
    image_batch, file_batch = tf.train.batch([image, file_id], batch_size=batch_size, 
                                num_threads=batch_threads, capacity=batch_capacity, 
                                allow_smaller_final_batch=True)
    
    
    return image_batch, file_batch