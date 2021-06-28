# 引入库文件
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models import resnet50, resnet101, resnet152, resnet34, resnet18
import config
from prepare_data import generate_datasets
import math
import os
import tensorflow_model_optimization as tfmot
import numpy as np
import tempfile
from StudentNet import StudentNet
import tensorflow_model_optimization as tfmot
import numpy as np
import tempfile


if __name__=='__main__':
    student_net = StudentNet()
    # student_net.build(input_shape=(config.image_height, config.image_width, config.channels))
    student_net.load_weights('student_net.h5')
    # student_net.summary()
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count,dataset_cat = generate_datasets()
    
    cnt = 0
    for images,labels in dataset_cat:
        if cnt==0:
            train_images = images
            train_labels = labels
            cnt+=1
            continue
        if cnt>=10:
            break
        cnt+=1
        train_images=tf.concat([train_images,images],0)
        train_labels=tf.concat([train_labels,labels],0)

    print(train_images.shape)
    print(train_labels.shape)
    input('wait here:press any key to continue')
            

  


   
    #-------------Define model for pruning.-----------------
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=50)
    }
    model_for_pruning = prune_low_magnitude(student_net, **pruning_params)
# `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    logdir = tempfile.mkdtemp()
    callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
    model_for_pruning.fit(train_images, train_labels,
                  batch_size=32, epochs=1, validation_split=0.1,
                  callbacks=callbacks)


    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    model_for_pruning.save('model_for_pruning.h5')

   
    #print('TF test accuracy:', model_accuracy)
    #print('Pruned TF test accuracy:', model_for_pruning_accuracy)
    #print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
    #print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
    #print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
    #print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (
    #    get_gzipped_model_size(quantized_and_pruned_tflite_file)))
