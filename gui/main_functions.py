import numpy as np
import datetime
import gc
import os
import pickle
import time
import tensorflow as tf
from keras_tuner import BayesianOptimization, Objective
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from workspace_setup import workspace_setup_utils
from preprocessing import preprocessing_utils
from model import hp_tuning_utils, training_utils, prediction_utils

def extract_data(data_dir, work_dir, output_log, start_btn, prog_bar, fwf_av):
    local_pathlist = workspace_setup_utils.create_working_directory(work_dir, output_log, fwf_av)
    workspace_setup_utils.unzip_all_datasets(data_dir, local_pathlist, start_btn, prog_bar, output_log, fwf_av)
    if fwf_av == True:
        data_unzipped_path, data_working_path, las_unzipped_path, fwf_unzipped_path, fpc_unzipped_path, las_working_path, fwf_working_path, img_working_path, metrics_path, config_data_path = workspace_setup_utils.get_workspace_path_mappings(local_pathlist, fwf_av)
        workspace_setup_utils.create_fpcs(fwf_unzipped_path, fpc_unzipped_path, output_log)
        workspace_setup_utils.extract_single_trees_from_fpc(fpc_unzipped_path, las_unzipped_path, las_working_path, fwf_working_path, output_log)
    else:
        pass
    return local_pathlist

def preprocess_data(local_pathlist, ssstest, capsel, growsel, elimper, maxpcscale, netpcsize, netimgsize, output_log, fwf_av):
    if fwf_av == True:
        selected_pointclouds_unaugmented, selected_fwf_pointclouds_unaugmented, pc_path_selection, fwf_path_selection, img_path_selection, metrics_path_selection = preprocessing_utils.traverse_user_specified_data(local_pathlist, capsel, growsel, output_log, fwf_av)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - AUGMENTING POINT CLOUDS" +'\n')
        output_log.configure(state="disabled")
        preprocessing_utils.augment_selection_fwf(selected_pointclouds_unaugmented, selected_fwf_pointclouds_unaugmented, elimper, maxpcscale, output_log)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING DEPTH IMAGES" +'\n')
        output_log.configure(state="disabled")
        preprocessing_utils.generate_colored_images(netimgsize, pc_path_selection, img_path_selection, output_log)
        selected_pointclouds_augmented, selected_fwf_pointclouds_augmented, selected_images_augmented = preprocessing_utils.get_user_specified_data_fwf(pc_path_selection, fwf_path_selection, img_path_selection, capsel, growsel)
        filtered_pointclouds, filtered_fwf_pointclouds, filtered_images = preprocessing_utils.filter_data_for_selection_fwf(selected_pointclouds_augmented, selected_fwf_pointclouds_augmented, selected_images_augmented, elimper)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - RESAMPLING POINT CLOUDS DENSITY BASED" +'\n')
        output_log.configure(state="disabled")
        pointclouds_for_resampling = [preprocessing_utils.load_point_cloud(file, output_log) for file in filtered_pointclouds]
        centered_points = preprocessing_utils.center_point_cloud(pointclouds_for_resampling)
        resampled_pointclouds = np.array([preprocessing_utils.resample_point_cloud_density_based(points, netpcsize, output_log) for points in centered_points])
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING FEATURES" +'\n')
        output_log.configure(state="disabled")
        combined_metrics_all = preprocessing_utils.generate_metrics_for_selected_pointclouds_fwf(filtered_pointclouds, filtered_fwf_pointclouds, metrics_path_selection, output_log)
        images_frontal, images_sideways = preprocessing_utils.match_images_with_pointclouds(filtered_pointclouds, filtered_images)
        images_front = np.asarray(images_frontal)
        images_side = np.asarray(images_sideways)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING TRAINING DATA" +'\n')
        output_log.configure(state="disabled")
        X_pc_train, X_pc_val, X_metrics_train, X_metrics_val, X_img_1_train, X_img_1_val, X_img_2_train, X_img_2_val, y_train, y_val, num_classes, label_dict = preprocessing_utils.generate_training_data(filtered_pointclouds, resampled_pointclouds, combined_metrics_all, images_front, images_side, ssstest, metrics_path_selection, elimper, output_log)
        return X_pc_train, X_pc_val, X_metrics_train, X_metrics_val, X_img_1_train, X_img_1_val, X_img_2_train, X_img_2_val, y_train, y_val, num_classes, label_dict
    else:
        selected_pointclouds_unaugmented, pc_path_selection, img_path_selection, metrics_path_selection = preprocessing_utils.traverse_user_specified_data(local_pathlist, capsel, growsel, output_log, fwf_av)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - AUGMENTING POINT CLOUDS" +'\n')
        output_log.configure(state="disabled")
        preprocessing_utils.augment_selection_fwf(selected_pointclouds_unaugmented, elimper, maxpcscale, output_log)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING DEPTH IMAGES" +'\n')
        output_log.configure(state="disabled")
        preprocessing_utils.generate_colored_images(netimgsize, pc_path_selection, img_path_selection, output_log)
        selected_pointclouds_augmented, selected_images_augmented = preprocessing_utils.get_user_specified_data(pc_path_selection, img_path_selection, capsel, growsel)
        filtered_pointclouds, filtered_images = preprocessing_utils.filter_data_for_selection(selected_pointclouds_augmented, selected_images_augmented, elimper)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - RESAMPLING POINT CLOUDS DENSITY BASED" +'\n')
        output_log.configure(state="disabled")
        pointclouds_for_resampling = [preprocessing_utils.load_point_cloud(file, output_log) for file in filtered_pointclouds]
        centered_points = preprocessing_utils.center_point_cloud(pointclouds_for_resampling)
        resampled_pointclouds = np.array([preprocessing_utils.resample_point_cloud_density_based(points, netpcsize, output_log) for points in centered_points])
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING FEATURES" +'\n')
        output_log.configure(state="disabled")
        combined_metrics_all = preprocessing_utils.generate_metrics_for_selected_pointclouds(filtered_pointclouds, metrics_path_selection, output_log)
        images_frontal, images_sideways = preprocessing_utils.match_images_with_pointclouds(filtered_pointclouds, filtered_images)
        images_front = np.asarray(images_frontal)
        images_side = np.asarray(images_sideways)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING TRAINING DATA" +'\n')
        output_log.configure(state="disabled")
        X_pc_train, X_pc_val, X_metrics_train, X_metrics_val, X_img_1_train, X_img_1_val, X_img_2_train, X_img_2_val, y_train, y_val, num_classes, label_dict = preprocessing_utils.generate_training_data(filtered_pointclouds, resampled_pointclouds, combined_metrics_all, images_front, images_side, ssstest, metrics_path_selection, elimper, output_log)
        return X_pc_train, X_pc_val, X_metrics_train, X_metrics_val, X_img_1_train, X_img_1_val, X_img_2_train, X_img_2_val, y_train, y_val, num_classes, label_dict
    
def preprocess_prediction_data(local_pathlist, capsel, growsel, elimper, maxpcscale, netpcsize, netimgsize, output_log, fwf_av):
    if fwf_av == True:
        selected_pointclouds_unaugmented, selected_fwf_pointclouds_unaugmented, pc_path_selection, fwf_path_selection, img_path_selection, metrics_path_selection = preprocessing_utils.traverse_user_specified_data(local_pathlist, capsel, growsel, output_log, fwf_av)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - AUGMENTING POINT CLOUDS" +'\n')
        output_log.configure(state="disabled")
        preprocessing_utils.augment_selection_fwf(selected_pointclouds_unaugmented, selected_fwf_pointclouds_unaugmented, elimper, maxpcscale, output_log)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING DEPTH IMAGES" +'\n')
        output_log.configure(state="disabled")
        preprocessing_utils.generate_colored_images(netimgsize, pc_path_selection, img_path_selection, output_log)
        selected_pointclouds_augmented, selected_fwf_pointclouds_augmented, selected_images_augmented = preprocessing_utils.get_user_specified_data_fwf(pc_path_selection, fwf_path_selection, img_path_selection, capsel, growsel)
        filtered_pointclouds, filtered_fwf_pointclouds, filtered_images = preprocessing_utils.filter_data_for_selection_fwf(selected_pointclouds_augmented, selected_fwf_pointclouds_augmented, selected_images_augmented, elimper)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - RESAMPLING POINT CLOUDS DENSITY BASED" +'\n')
        output_log.configure(state="disabled")
        pointclouds_for_resampling = [preprocessing_utils.load_point_cloud(file, output_log) for file in filtered_pointclouds]
        centered_points = preprocessing_utils.center_point_cloud(pointclouds_for_resampling)
        resampled_pointclouds = np.array([preprocessing_utils.resample_point_cloud_density_based(points, netpcsize, output_log) for points in centered_points])
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING FEATURES" +'\n')
        output_log.configure(state="disabled")
        combined_metrics_all = preprocessing_utils.generate_metrics_for_selected_pointclouds_fwf(filtered_pointclouds, filtered_fwf_pointclouds, metrics_path_selection, output_log)
        images_frontal, images_sideways = preprocessing_utils.match_images_with_pointclouds(filtered_pointclouds, filtered_images)
        images_front = np.asarray(images_frontal)
        images_side = np.asarray(images_sideways)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING TRAINING DATA" +'\n')
        output_log.configure(state="disabled")
        X_pc, X_metrics, X_img_1, X_img_2, y, onehot_to_label_dict = prediction_utils.generate_prediction_data(filtered_pointclouds, resampled_pointclouds, combined_metrics_all, images_front, images_side, metrics_path_selection, elimper, output_log)
        return X_pc, X_metrics, X_img_1, X_img_2, y, onehot_to_label_dict, filtered_pointclouds
    else:
        selected_pointclouds_unaugmented, pc_path_selection, img_path_selection, metrics_path_selection = preprocessing_utils.traverse_user_specified_data(local_pathlist, capsel, growsel, output_log, fwf_av)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - AUGMENTING POINT CLOUDS" +'\n')
        output_log.configure(state="disabled")
        preprocessing_utils.augment_selection_fwf(selected_pointclouds_unaugmented, elimper, maxpcscale, output_log)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING DEPTH IMAGES" +'\n')
        output_log.configure(state="disabled")
        preprocessing_utils.generate_colored_images(netimgsize, pc_path_selection, img_path_selection, output_log)
        selected_pointclouds_augmented, selected_images_augmented = preprocessing_utils.get_user_specified_data(pc_path_selection, img_path_selection, capsel, growsel)
        filtered_pointclouds, filtered_images = preprocessing_utils.filter_data_for_selection(selected_pointclouds_augmented, selected_images_augmented, elimper)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - RESAMPLING POINT CLOUDS DENSITY BASED" +'\n')
        output_log.configure(state="disabled")
        pointclouds_for_resampling = [preprocessing_utils.load_point_cloud(file, output_log) for file in filtered_pointclouds]
        centered_points = preprocessing_utils.center_point_cloud(pointclouds_for_resampling)
        resampled_pointclouds = np.array([preprocessing_utils.resample_point_cloud_density_based(points, netpcsize, output_log) for points in centered_points])
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING FEATURES" +'\n')
        output_log.configure(state="disabled")
        combined_metrics_all = preprocessing_utils.generate_metrics_for_selected_pointclouds(filtered_pointclouds, metrics_path_selection, output_log)
        images_frontal, images_sideways = preprocessing_utils.match_images_with_pointclouds(filtered_pointclouds, filtered_images)
        images_front = np.asarray(images_frontal)
        images_side = np.asarray(images_sideways)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - GENERATING TRAINING DATA" +'\n')
        output_log.configure(state="disabled")
        X_pc, X_metrics, X_img_1, X_img_2, y, onehot_to_label_dict = prediction_utils.generate_prediction_data(filtered_pointclouds, resampled_pointclouds, combined_metrics_all, images_front, images_side, metrics_path_selection, elimper, output_log)
        return X_pc, X_metrics, X_img_1, X_img_2, y, onehot_to_label_dict, filtered_pointclouds

def perform_hp_tuning(model_dir, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, bsize, netpcsize, netimgsize, num_classes, output_log):
    point_cloud_shape = (netpcsize, 3)
    image_shape = (netimgsize, netimgsize, 3)
    metrics_shape = (X_metrics_train.shape[1],)
    batch_size = bsize
    num_hp_epochs = 15
    num_hp_trials = 15  
    os.chdir(model_dir)
    tf.keras.backend.clear_session()
    X_img_1_train, X_img_2_train, X_pc_train, X_metrics_train = hp_tuning_utils.normalize_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train)
    X_img_1_val, X_img_2_val, X_pc_val, X_metrics_val = hp_tuning_utils.normalize_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val)
    hp_tuning_utils.check_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train)
    hp_tuning_utils.check_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val)
    corruption_found = hp_tuning_utils.check_label_corruption(y_train)
    if not corruption_found:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - INFO - No corruption found in one-hot encoded labels!" +'\n')
        output_log.configure(state="disabled")
    corruption_found = hp_tuning_utils.check_label_corruption(y_val)
    if not corruption_found:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - INFO - No corruption found in one-hot encoded labels!" +'\n')
        output_log.configure(state="disabled")
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - INFO - Distribution in training data: {hp_tuning_utils.get_class_distribution(y_train)}" +'\n')
    output_log.insert("1.0", f"{now_formatted} - INFO - Distribution in validation data: {hp_tuning_utils.get_class_distribution(y_val)}" +'\n')
    output_log.configure(state="disabled")
    train_gen = hp_tuning_utils.DataGenerator(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, batch_size)
    val_gen = hp_tuning_utils.DataGenerator(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, batch_size)
    train_gen.on_epoch_end()
    val_gen.on_epoch_end()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dir_name = f'hp-tuning-mmtscnet-{timestamp}'
    tuner = BayesianOptimization(
        hp_tuning_utils.CombinedModel(point_cloud_shape, image_shape, metrics_shape, num_classes, netpcsize),
        objective=Objective("val_macro_f1", direction="max"),
        max_trials=num_hp_trials,
        max_retries_per_trial=3,
        max_consecutive_failed_trials=3,
        directory=dir_name,
        project_name='tree_classification'
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.985, patience=5, min_lr=1e-7)
    degrade_lr = LearningRateScheduler(hp_tuning_utils.scheduler)
    macro_f1_callback = hp_tuning_utils.MacroF1ScoreCallback(validation_data=val_gen, batch_size=batch_size)
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - INFO - Commencing hyperparameter-tuning for {num_hp_trials} trials with {num_hp_epochs} epochs!" +'\n')
    output_log.configure(state="disabled")
    tuner.search(train_gen,
                epochs=num_hp_epochs,
                validation_data=val_gen,
                class_weight=hp_tuning_utils.generate_class_weights(y_train),
                callbacks=[reduce_lr, degrade_lr, macro_f1_callback])
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    combined_model = hp_tuning_utils.CombinedModel(point_cloud_shape, image_shape, metrics_shape, num_classes, netpcsize)
    untrained_model = combined_model.get_untrained_model(best_hyperparameters)
    untrained_model.summary()
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - INFO - Finished hp-tuning!" +'\n')
    output_log.configure(state="disabled")
    gc.collect()
    return untrained_model

def perform_training(model, bsz, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, modeldir, label_dict, output_log):
    tf.keras.utils.set_random_seed(812)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7.5e-6),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="pr_curve", curve="PR"), tf.keras.metrics.PrecisionAtRecall(0.85, name="pr_at_rec"), tf.keras.metrics.RecallAtPrecision(0.85, name="rec_at_pr")]
    )
    model._name="MMTSCNet_V2"
    model.summary()
    os.chdir(modeldir)
    tf.keras.backend.clear_session()
    X_img_1_train, X_img_2_train, X_pc_train, X_metrics_train = hp_tuning_utils.normalize_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train)
    X_img_1_val, X_img_2_val, X_pc_val, X_metrics_val = hp_tuning_utils.normalize_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val)
    hp_tuning_utils.check_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train)
    hp_tuning_utils.check_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val)
    corruption_found = hp_tuning_utils.check_label_corruption(y_train)
    if not corruption_found:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - INFO - No corruption found in one-hot encoded labels!" +'\n')
        output_log.configure(state="disabled")
    corruption_found = hp_tuning_utils.check_label_corruption(y_val)
    if not corruption_found:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - INFO - No corruption found in one-hot encoded labels!" +'\n')
        output_log.configure(state="disabled")
    train_gen = hp_tuning_utils.DataGenerator(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, bsz)
    val_gen = hp_tuning_utils.DataGenerator(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, bsz)
    train_gen.on_epoch_end()
    val_gen.on_epoch_end()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.985, patience=3, min_lr=2e-7)
    degrade_lr = tf.keras.callbacks.LearningRateScheduler(hp_tuning_utils.scheduler)
    macro_f1_callback = hp_tuning_utils.MacroF1ScoreCallback(validation_data=val_gen, batch_size=bsz)
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - INFO - Distribution in training data: {hp_tuning_utils.get_class_distribution(y_train)}" +'\n')
    output_log.insert("1.0", f"{now_formatted} - INFO - Distribution in validation data: {hp_tuning_utils.get_class_distribution(y_val)}" +'\n')
    output_log.configure(state="disabled")
    history = model.fit(
        train_gen,
        epochs=300,
        validation_data=val_gen,
        class_weight=hp_tuning_utils.generate_class_weights(y_train),
        callbacks=[early_stopping, reduce_lr, degrade_lr, macro_f1_callback],
        verbose=1
    )
    hp_tuning_utils.plot_and_save_history(history, modeldir, 0)
    history = history.history
    with open('history.pkl', 'wb') as f:
        pickle.dump(history, f)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_file_path = f'MMTSCNET_{timestamp}_TRAINED'
    if os.path.exists(model_file_path):
        os.remove(model_file_path)
    try:
        model.save(model_file_path, save_format="keras")
    except:
        pass
    predictions = model.predict([X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val], 1, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_pred_real = hp_tuning_utils.map_onehot_to_real(y_pred, label_dict)
    y_true_real = hp_tuning_utils.map_onehot_to_real(y_val, label_dict)
    hp_tuning_utils.plot_conf_matrix(y_true_real, y_pred_real, modeldir, model_file_path)
    model.summary()
    model_path = training_utils.get_trained_model_folder(modeldir)
    model_loaded_test = training_utils.load_trained_model_from_folder(model_path)
    model_loaded_test.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7.5e-6),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="pr_curve", curve="PR"), tf.keras.metrics.PrecisionAtRecall(0.85, name="pr_at_rec"), tf.keras.metrics.RecallAtPrecision(0.85, name="rec_at_pr")]
    )
    model_loaded_test.summary()
    gc.collect()
    return model_loaded_test

def predict_for_custom_data(model, X_pc_pred, X_image_f_pred, X_image_s_pred, X_metrics_pred, y_true, label_dict, workspace_pathlist, work_dir, filtered_pointclouds, output_log):
    #----WIP----#


    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - INFO - Saved classified pointclouds @ NONE!" +'\n')
    output_log.configure(state="disabled")