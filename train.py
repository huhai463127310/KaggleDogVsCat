# -*- coding: utf-8 -*-

"""
训练脚本
"""
import inspect
import time
import os
import shutil


import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.contrib.framework import assign_from_checkpoint_fn
from lib import inception_resnet_v2

from pre_process import read_image
from pre_process import config_log

def accuracy_fn(predictions, labels, name='accuracy'):
    """
    精确度计算函数
    """
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32), name=name)

def filter_train_variables_by_layer(names):
    """
    根据名字过滤训练变量
    """
    variables = list()
    for v in tf.trainable_variables():
        for name in names:
            if v.name.find(name) == 0:
                variables.append(v)
    return variables


def model_vgg_16(inputs, scope, num_classes=2, is_training=True):    
    return  nets.vgg.vgg_16(inputs, num_classes=num_classes, scope=scope, is_training=is_training)

model_vgg_16.train_layer = ['vgg_16/fc8']
model_vgg_16.scope = 'vgg_16'
model_vgg_16.arg_scope = None
model_vgg_16.pre_trained_model = 'pre_train_model/vgg_16.ckpt'
model_vgg_16.image_height = 224
model_vgg_16.image_weight = 224

"""
vgg_16 end_points

vgg_16/vgg_16/conv1/conv1_1
vgg_16/vgg_16/conv1/conv1_2
vgg_16/vgg_16/pool1
vgg_16/vgg_16/conv2/conv2_1
vgg_16/vgg_16/conv2/conv2_2
vgg_16/vgg_16/pool2
vgg_16/vgg_16/conv3/conv3_1
vgg_16/vgg_16/conv3/conv3_2
vgg_16/vgg_16/conv3/conv3_3
vgg_16/vgg_16/pool3
vgg_16/vgg_16/conv4/conv4_1
vgg_16/vgg_16/conv4/conv4_2
vgg_16/vgg_16/conv4/conv4_3
vgg_16/vgg_16/pool4
vgg_16/vgg_16/conv5/conv5_1
vgg_16/vgg_16/conv5/conv5_2
vgg_16/vgg_16/conv5/conv5_3
vgg_16/vgg_16/pool5
vgg_16/vgg_16/fc6
vgg_16/vgg_16/fc7
vgg_16/vgg_16/fc8
vgg_16/fc8
"""

def model_inception_resnet_v2(inputs, scope, num_classes=2, is_training=True):            
    return  inception_resnet_v2.inception_resnet_v2(inputs, num_classes=num_classes, scope=scope, is_training=is_training, dropout_keep_prob=0.5)

model_inception_resnet_v2.train_layer = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
model_inception_resnet_v2.scope = "InceptionResnetV2"
model_inception_resnet_v2.arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
model_inception_resnet_v2.pre_trained_model = 'pre_train_model/inception_resnet_v2_2016_08_30.ckpt'
model_inception_resnet_v2.image_height = 299
model_inception_resnet_v2.image_weight = 299

"""
inception_resnet_v2 end_points

Conv2d_1a_3x3
Conv2d_2a_3x3
Conv2d_2b_3x3
MaxPool_3a_3x3
Conv2d_3b_1x1
Conv2d_4a_3x3
MaxPool_5a_3x3
Mixed_5b
Mixed_6a
PreAuxLogits
Mixed_7a
Conv2d_7b_1x1
AuxLogits
global_pool
PreLogitsFlatten
Logits
Predictions
"""

def train(model_fn, pre_trained_model, train_log_dir, scope, arg_scope, train_layer, 
          epochs=1, steps=None, learning_rate=0.001, num_classes=2, decay_steps=1000, decay_rate=0.8, save_interval_secs=300,
          image_h=224, image_w=224, batch_size=32, batch_threads=10, log_every_n_steps = 10,
          train_image_dir = "input/train", validation_image_dir = "input/validation", file_ext_name=".jpg",
          restore_full_layer=False, lock_layer=True):
    
    batch_capacity = batch_size * 2
    train_image_path = os.path.join(train_image_dir, "*" + file_ext_name)
    validation_image_path = os.path.join(validation_image_dir, "*" + file_ext_name)
    
    tf.logging.info("train_image_path={} validation_image_path={}".format(train_image_path, validation_image_path))
    
    # 计算训练图片数
    train_image_nums = len([f for f in os.listdir(train_image_dir) if os.path.splitext(f)[1].lower() == file_ext_name.lower()])

    steps_per_epoch = train_image_nums // batch_size
    max_step = steps if steps is not None else steps_per_epoch * epochs
    epochs_ = max_step // steps_per_epoch + (1 if max_step % steps_per_epoch != 0 else 0) if steps is not None else epochs
    
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  

    with tf.Graph().as_default():

        image_train, label_train, name_train = read_image(
            train_image_path, image_h, image_w, epochs=None, batch_size=batch_size, batch_threads=batch_threads, 
            batch_capacity=batch_capacity, data_augmentation=True)
    
        image_val, label_val, name_val = read_image(
            validation_image_path, image_h, image_w, epochs=None, batch_size=batch_size*5, batch_threads=batch_threads, 
            batch_capacity=batch_capacity*5)
    
        with tf.variable_scope(scope) as scope_:
            if arg_scope:
                with slim.arg_scope(arg_scope):
                    predictions, end_points = model_fn(image_train, num_classes=num_classes, scope=scope_)
                    scope_.reuse_variables()
                    predictions_val, end_points_val = model_fn(image_val, num_classes=num_classes, 
                                                               scope=scope_, is_training=False)
            else:
                predictions, end_points = model_fn(image_train, num_classes=num_classes, scope=scope_)
                scope_.reuse_variables()
                predictions_val, end_points_val = model_fn(image_val, num_classes=num_classes, 
                                                           scope=scope_, is_training=False)
                
#         for v in end_points:
#             print(v)
        
        excludes = train_layer if not restore_full_layer else None
        variables_to_restore = slim.get_variables_to_restore(exclude=excludes)
        init_fn = assign_from_checkpoint_fn(pre_trained_model, variables_to_restore)        

        # Specify the loss function:
        loss = tf.losses.softmax_cross_entropy(label_train, predictions)
        total_loss = slim.losses.get_total_loss()
        
        loss_val = tf.losses.softmax_cross_entropy(label_val, predictions_val)
        
        accuracy = accuracy_fn(predictions, label_train)
        accuracy_val = accuracy_fn(predictions_val, label_val, "accuracy_val")

        tf.summary.scalar('train/loss', loss)
        tf.summary.scalar('train/accuracy', accuracy)
        tf.summary.image('train/inputs', tf.reshape(image_train, [-1, image_h, image_w, 3]), 5)
        
        tf.summary.scalar('validation/loss', loss_val)
        tf.summary.scalar('validation/accuracy', accuracy_val)
        tf.summary.image('validation/inputs', tf.reshape(image_val, [-1, image_h, image_w, 3]), 5)
        
        global_step = slim.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)

        # Specify the optimization scheme:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        variables_to_train = filter_train_variables_by_layer(train_layer) if lock_layer else None
        if lock_layer:
            assert variables_to_train, "no variables to train, train_layer={}".format(train_layer)
        train_tensor = slim.learning.create_train_op(total_loss, optimizer, global_step=global_step, 
                variables_to_train=variables_to_train)
        # Actually runs training.
        
        start_time = time.time()
        
        def train_step(sess, train_op, global_step, train_step_kwargs): 
            global_step_value, accuracy_value, loss_value, accuracy_validation_value, loss_val_value, learning_rate_value = \
                sess.run([global_step, accuracy, loss, accuracy_val, loss_val, learning_rate]) # , end_points['Logits']
            if global_step_value % log_every_n_steps == 0 or global_step_value >= max_step - 1:
                tf.logging.info("global_step = {}/{} epoch = {}/{} accuracy = {:.5f} loss = {:.5f} accuracy_val = {:.5f} loss_val = {:.5f} learning_rate = {:.5f} time_elipse = {:.2f} s".format(
                    global_step_value + 1, max_step, global_step_value // steps_per_epoch + 1, epochs_, accuracy_value, loss_value, accuracy_validation_value, loss_val_value, learning_rate_value, time.time() - start_time))
            
#             if global_step_value >= max_step - 1:
#                 tf.logging.info("logits_shape={}".format(logits_value.shape))
            return slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)

        slim.learning.train(train_tensor, train_log_dir, 
                            init_fn=init_fn, train_step_fn=train_step, global_step=global_step, 
                            log_every_n_steps=log_every_n_steps, save_summaries_secs=60, 
                            save_interval_secs=save_interval_secs, number_of_steps=max_step, session_config=config)


if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_string("model", "", "model:vgg_16, inception_resnet_v2")
    flags.DEFINE_integer("epochs", 1, "epoches")
    flags.DEFINE_integer("steps", None, "max steps")
    flags.DEFINE_integer("decay_steps", 100000, "learning rate decay steps")
    flags.DEFINE_float("decay_rate", 0.996, "decay rate for learning rate")
    flags.DEFINE_float("learning_rate", 0.0001, "learning_rate")
    
    args = flags.FLAGS
    
    if args.model == "":
        print("missing parameter 'model'")
        exit(1)
        
    train_func_name = "model_" + args.model
    lv = locals()

    if train_func_name not in lv or not inspect.isfunction(lv[train_func_name]):
        print("model {} not supported".format(args.model))
        exit(1)
        
    train_log_dir = "logs/" + args.model
    
    if tf.gfile.Exists(train_log_dir):
        print("log dir exists: {}".format(train_log_dir))
        exit(1)
        
    tf.gfile.MakeDirs(train_log_dir)

    config_log(os.path.join(train_log_dir, "console.log"))
    start_time = time.time()
    tf.logging.info("start train {} learning_rate={} epochs={} steps={} decay_steps={} decay_rate={}".format(
        args.model, args.learning_rate, args.epochs, args.steps, args.decay_steps, args.decay_rate))

    tf.set_random_seed(10)
    
    model_fn = lv[train_func_name]
    train(model_fn, model_fn.pre_trained_model, train_log_dir, 
          scope=model_fn.scope, arg_scope=model_fn.arg_scope, train_layer=model_fn.train_layer, 
          epochs=args.epochs, steps=args.steps, image_h=model_fn.image_height, image_w=model_fn.image_weight,
          learning_rate=args.learning_rate, decay_steps=args.decay_steps, decay_rate=args.decay_rate)

    end_time = time.time()
    tf.logging.info("train finished. cost {:.2f} minutes".format((end_time - start_time)/60))
    