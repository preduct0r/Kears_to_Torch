# from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>DESIRED_ACCURACY):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

# model_file = os.path.join(models_dir, 'rawwav1d_{}_{}_{}.h5'.format(feature_shape[0], to_expand, 1))
# checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=show_train_info, save_best_only=True)
# reduce_lr = LearningRateScheduler(linear_decay_lr(5, config.max_epochs))
# stopper = EarlyStopping(patience=20)
