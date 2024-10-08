from functionalities import workspace_setup, preprocessing, model_utils
import numpy as np
import os
import tensorflow as tf
import gc
import time
from keras_tuner import BayesianOptimization, Objective
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import datetime

def extract_data(data_dir, work_dir, fwf_av, capsel, growsel, start_btn, progbar, output_log):
    """
    Main utility function for i/o ops before preprocessing.

    Args:
    data_dir: User-specified source data directory.
    work_dir: User-specified working directory.
    fwf_av: True/False - Presence of FWF data.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.

    Return:
    full_pathlist: List of all paths available to MMTSCNet.
    """
    local_pathlist = workspace_setup.create_working_directory(work_dir, fwf_av, output_log)
    workspace_setup.unzip_all_datasets(data_dir, local_pathlist, fwf_av, start_btn, progbar, output_log)
    if fwf_av == True:
        workspace_setup.create_fpcs(local_pathlist[2], local_pathlist[3])
        full_pathlist = workspace_setup.create_config_directory(local_pathlist, capsel, growsel, fwf_av, output_log)
        workspace_setup.extract_single_trees_from_fpc(local_pathlist[3], local_pathlist[1], full_pathlist[6], full_pathlist[7], capsel, growsel, output_log)
        return full_pathlist
    else:
        full_pathlist = workspace_setup.create_config_directory(local_pathlist, capsel, growsel, fwf_av, output_log)
        return full_pathlist

def preprocess_data(full_pathlist, ssstest, capsel, growsel, elimper, maxpcscale, netpcsize, netimgsize, fwf_av, start_btn, progbar, output_log):
    """
    Main utility function for the preprocessing.

    Args:
    full_pathlist: List of all paths used for MMTSCNet.
    ssstest: Train/Test split ratio.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.
    elimper: Threshold for the elimination of underrepresented species.
    maxpcscale: Maximum scaling to apply during augmentation.
    netpcsize: Number of points to resample point clouds to.
    netimgsize: Image size in Pixels (224).
    fwf_av: True/False - Presence of FWF data.

    Returns:
    X_pc_train: Training point clouds.
    X_pc_val: Validation point clouds.
    X_metrics_train: Training numerical features.
    X_metrics_val: Validation numerical features.
    X_img_1_train: Frontal validation images.
    X_img_1_val: Frontal validation images.
    X_img_2_train: Sideways training images.
    X_img_2_val: Sideways validation images.
    y_train: Training labels.
    y_val: Validation labels.
    num_classes: Number of classes present in the training dataset.
    label_dict: Dictionary to translate one-hot encoded labels to textual labels.
    """
    if fwf_av == True:
        unaugmented_regular_pointclouds = preprocessing.select_pointclouds(full_pathlist[6])
        unaugmented_fwf_pointclouds = preprocessing.select_pointclouds(full_pathlist[7])
        preprocessing.augment_selection_fwf(unaugmented_regular_pointclouds, unaugmented_fwf_pointclouds, elimper, maxpcscale, full_pathlist[6], full_pathlist[7], netpcsize, capsel, output_log)
        preprocessing.generate_colored_images(netimgsize, full_pathlist[6], full_pathlist[8], output_log)
        selected_pointclouds_augmented, selected_fwf_pointclouds_augmented, selected_images_augmented = preprocessing.get_user_specified_data_fwf(full_pathlist[6], full_pathlist[7], full_pathlist[8], capsel, growsel)
        filtered_pointclouds, filtered_fwf_pointclouds, filtered_images = preprocessing.filter_data_for_selection_fwf(selected_pointclouds_augmented, selected_fwf_pointclouds_augmented, selected_images_augmented, elimper)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Resampling point clouds density based!" +'\n')
        output_log.configure(state="disabled")
        pointclouds_for_resampling = [preprocessing.load_point_cloud(file) for file in filtered_pointclouds]
        centered_points = preprocessing.center_point_cloud(pointclouds_for_resampling)
        resampled_pointclouds = np.array([preprocessing.resample_point_cloud_density_based(points, netpcsize, output_log) for points in centered_points])
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Generating metrics for point clouds!" +'\n')
        output_log.configure(state="disabled")
        combined_metrics_all = preprocessing.generate_metrics_for_selected_pointclouds_fwf(filtered_pointclouds, filtered_fwf_pointclouds, full_pathlist[9], capsel, growsel, output_log)
        images_frontal, images_sideways = preprocessing.match_images_with_pointclouds(filtered_pointclouds, filtered_images)
        images_front = np.asarray(images_frontal)
        images_side = np.asarray(images_sideways)
        X_pc_train, X_pc_val, X_metrics_train, X_metrics_val, X_img_1_train, X_img_1_val, X_img_2_train, X_img_2_val, y_train, y_val, num_classes, label_dict = preprocessing.generate_training_data(output_log, capsel, growsel, filtered_pointclouds, resampled_pointclouds, combined_metrics_all, images_front, images_side, ssstest, full_pathlist[9], 0.003)
        return X_pc_train, X_pc_val, X_metrics_train, X_metrics_val, X_img_1_train, X_img_1_val, X_img_2_train, X_img_2_val, y_train, y_val, num_classes, label_dict
    else:
        unaugmented_regular_pointclouds = preprocessing.select_pointclouds(full_pathlist[4])
        preprocessing.augment_selection(unaugmented_regular_pointclouds, elimper, maxpcscale, full_pathlist[4], netpcsize, capsel, output_log)
        preprocessing.generate_colored_images(netimgsize, full_pathlist[4], full_pathlist[5], output_log)
        selected_pointclouds_augmented, selected_images_augmented = preprocessing.get_user_specified_data(full_pathlist[4], full_pathlist[5], capsel, growsel)
        filtered_pointclouds, filtered_images = preprocessing.filter_data_for_selection(selected_pointclouds_augmented, selected_images_augmented, elimper)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Resampling point clouds density based!" +'\n')
        output_log.configure(state="disabled")
        pointclouds_for_resampling = [preprocessing.load_point_cloud(file) for file in filtered_pointclouds]
        centered_points = preprocessing.center_point_cloud(pointclouds_for_resampling)
        resampled_pointclouds = np.array([preprocessing.resample_point_cloud_density_based(points, netpcsize, output_log) for points in centered_points])
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Generating metrics for point clouds!" +'\n')
        output_log.configure(state="disabled")
        combined_metrics_all = preprocessing.generate_metrics_for_selected_pointclouds(filtered_pointclouds, full_pathlist[6], capsel, growsel, output_log)
        images_frontal, images_sideways = preprocessing.match_images_with_pointclouds(filtered_pointclouds, filtered_images)
        images_front = np.asarray(images_frontal)
        images_side = np.asarray(images_sideways)
        X_pc_train, X_pc_val, X_metrics_train, X_metrics_val, X_img_1_train, X_img_1_val, X_img_2_train, X_img_2_val, y_train, y_val, num_classes, label_dict = preprocessing.generate_training_data(output_log, capsel, growsel, filtered_pointclouds, resampled_pointclouds, combined_metrics_all, images_front, images_side, ssstest, full_pathlist[6], 0.003)
        return X_pc_train, X_pc_val, X_metrics_train, X_metrics_val, X_img_1_train, X_img_1_val, X_img_2_train, X_img_2_val, y_train, y_val, num_classes, label_dict
    
def perform_hp_tuning(model_dir, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, bsize, netpcsize, netimgsize, num_classes, capsel, growsel, start_btn, progbar, output_log):
    """
    Main utility function for the hyperparameter tuning process.

    Args:
    model_dir: Filepath to model saving destination.
    X_pc_train: Training point clouds.
    X_pc_val: Validation point clouds.
    X_metrics_train: Training numerical features.
    X_metrics_val: Validation numerical features.
    X_img_1_train: Frontal validation images.
    X_img_1_val: Frontal validation images.
    X_img_2_train: Sideways training images.
    X_img_2_val: Sideways validation images.
    y_train: Training labels.
    y_val: Validation labels.
    bsize: User-specified batch size.
    netpcsize: Number of points to resample point clouds to.
    netimgsize: Image size in Pixels (224).
    num_classes: Number of classes present in the training dataset.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.
    fwf_av: True/False - Presence of FWF data.

    Returns:
    untrained_model: Keras model instance of MMTSCNet with tuned hyperparameters.
    """
    point_cloud_shape = (netpcsize, 3)
    image_shape = (netimgsize, netimgsize, 3)
    metrics_shape = (X_metrics_train.shape[1],)
    batch_size = bsize
    num_hp_epochs = 15
    num_hp_trials = 12
    os.chdir(model_dir)
    tf.keras.backend.clear_session()
    X_img_1_train, X_img_2_train, X_pc_train, X_metrics_train = model_utils.normalize_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train)
    X_img_1_val, X_img_2_val, X_pc_val, X_metrics_val = model_utils.normalize_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val)
    model_utils.check_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train)
    model_utils.check_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val)
    corruption_found = model_utils.check_label_corruption(y_train)
    if not corruption_found:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - No corruption found in one-hot encoded training labels!" +'\n')
        output_log.configure(state="disabled")
    corruption_found = model_utils.check_label_corruption(y_val)
    if not corruption_found:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - No corruption found in one-hot encoded testing labels!" +'\n')
        output_log.configure(state="disabled")
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - STATUS - Distribution in training data: {model_utils.get_class_distribution(y_train)}!" +'\n')
    output_log.configure(state="disabled")
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - STATUS - Distribution in testing data: {model_utils.get_class_distribution(y_val)}!" +'\n')
    output_log.configure(state="disabled")
    train_gen = model_utils.DataGenerator(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, batch_size)
    val_gen = model_utils.DataGenerator(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, batch_size)
    train_gen.on_epoch_end()
    val_gen.on_epoch_end()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dir_name = f'hp-tuning_{capsel}_{growsel}_{timestamp}'
    tuner = BayesianOptimization(
        model_utils.CombinedModel(point_cloud_shape, image_shape, metrics_shape, num_classes, netpcsize),
        objective=Objective("val_custom_metric", direction="max"),
        max_trials=num_hp_trials,
        max_retries_per_trial=3,
        max_consecutive_failed_trials=3,
        directory='hp-tuning_ULS_LEAF-ON_20240713-233054',
        project_name='tree_classification'
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.985, patience=5, min_lr=1e-7)
    degrade_lr = LearningRateScheduler(model_utils.scheduler)
    macro_f1_callback = model_utils.MacroF1ScoreCallback(validation_data=val_gen, batch_size=batch_size)
    custom_scoring_callback = model_utils.WeightedResultsCallback(validation_data=val_gen, batch_size=batch_size)
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - STATUS - Commencing hyperparameter-tuning for {num_hp_trials} trials with {num_hp_epochs} epochs!!" +'\n')
    output_log.configure(state="disabled")
    tuner.search(train_gen,
                epochs=num_hp_epochs,
                validation_data=val_gen,
                class_weight=model_utils.generate_class_weights(y_train),
                callbacks=[reduce_lr, degrade_lr, macro_f1_callback, custom_scoring_callback])
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    combined_model = model_utils.CombinedModel(point_cloud_shape, image_shape, metrics_shape, num_classes, netpcsize)
    untrained_model = combined_model.get_untrained_model(best_hyperparameters)
    untrained_model.summary()
    gc.collect()
    return untrained_model

def perform_training(model, bsz, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, modeldir, label_dict, capsel, growsel, start_btn, progbar, output_log):
    """
    Main utility function for the training process.

    Args:
    model: Untrained tuned model instance.
    bsz: User-specified batch size.
    X_pc_train: Training point clouds.
    X_pc_val: Validation point clouds.
    X_metrics_train: Training numerical features.
    X_metrics_val: Validation numerical features.
    X_img_1_train: Frontal validation images.
    X_img_1_val: Frontal validation images.
    X_img_2_train: Sideways training images.
    X_img_2_val: Sideways validation images.
    y_train: Training labels.
    y_val: Validation labels.
    model_dir: Target directory for model saving.
    label_dict: Dictionary to translate one-hot encoded labels to textual labels.
    netpcsize: Number of points to resample point clouds to.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.
    fwf_av: True/False - Presence of FWF data.

    Returns:
    trained_model: Keras model instance of the trained MMTSCNet.
    """
    y_pred_val = y_val
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
    X_img_1_train, X_img_2_train, X_pc_train, X_metrics_train = model_utils.normalize_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train)
    X_img_1_val, X_img_2_val, X_pc_val, X_metrics_val = model_utils.normalize_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val)
    model_utils.check_data(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train)
    model_utils.check_data(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val)
    corruption_found = model_utils.check_label_corruption(y_train)
    if not corruption_found:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - No corruption found in one-hot encoded training labels!" +'\n')
        output_log.configure(state="disabled")
    corruption_found = model_utils.check_label_corruption(y_val)
    if not corruption_found:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - No corruption found in one-hot encoded testing labels!" +'\n')
        output_log.configure(state="disabled")
    train_gen = model_utils.DataGenerator(X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, bsz)
    val_gen = model_utils.DataGenerator(X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, bsz)
    train_gen.on_epoch_end()
    val_gen.on_epoch_end()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.985, patience=3, min_lr=2e-7)
    degrade_lr = tf.keras.callbacks.LearningRateScheduler(model_utils.scheduler)
    macro_f1_callback = model_utils.MacroF1ScoreCallback(validation_data=val_gen, batch_size=bsz)
    custom_scoring_callback = model_utils.WeightedResultsCallback(validation_data=val_gen, batch_size=bsz)
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - STATUS - Distribution in training data: {model_utils.get_class_distribution(y_train)}!" +'\n')
    output_log.configure(state="disabled")
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - STATUS - Distribution in testing data: {model_utils.get_class_distribution(y_val)}!" +'\n')
    output_log.configure(state="disabled")
    history = model.fit(
        train_gen,
        epochs=300,
        validation_data=val_gen,
        class_weight=model_utils.generate_class_weights(y_train),
        callbacks=[early_stopping, reduce_lr, degrade_lr, macro_f1_callback, custom_scoring_callback],
        verbose=1
    )
    plot_path = model_utils.plot_and_save_history(history, modeldir, capsel, growsel)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_file_path = f'MMTSCNET_{capsel}_{growsel}_{timestamp}_TRAINED'
    if os.path.exists(model_file_path):
        os.remove(model_file_path)
    try:
        model.save(model_file_path, save_format="keras")
    except:
        pass
    predictions = model.predict([X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val], batch_size=16, verbose=1)
    y_pred_real = model_utils.map_onehot_to_real(predictions, label_dict)
    y_true_real = model_utils.map_onehot_to_real(y_pred_val, label_dict)
    model_utils.plot_conf_matrix(y_true_real, y_pred_real, modeldir, plot_path, label_dict)
    model_utils.plot_best_epoch_metrics(history, plot_path)
    model.summary()
    model_path = model_utils.get_trained_model_folder(modeldir, capsel, growsel)
    model_loaded_test = model_utils.load_trained_model_from_folder(model_path)
    gc.collect()
    return model_loaded_test

def predict_for_custom_data(pretrained_model, work_dir, netimgsize, netpcsize, capsel, growsel, elimper, fwf_av, data_dir, model_dir, start_btn, progbar, output_log):
    """
    Main utility function for the prediction on custom data.

    Args:
    pretrained_model: Trained and tuned model instance.
    work_dir: Filepath to the working directory.
    netimgsize: Input image size forthe image processing branches (224).
    netpcsize: Number of points to resample point clouds to.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.
    fwf_av: True/False - Presence of FWF data.
    elimper: Elimination threshold for underrepresented species.
    data_dir: Source data directory filepath.
    model_dir: Model directory filepath.
    """
    if fwf_av == True:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Extracting data!" +'\n')
        output_log.configure(state="disabled")
        full_pathlist = workspace_setup.extract_data_for_predictions(data_dir, work_dir, fwf_av, capsel, growsel, output_log)
        preprocessing.generate_colored_images(netimgsize, full_pathlist[6], full_pathlist[8])
        selected_pointclouds, selected_fwf_pointclouds, selected_images_augmented = preprocessing.get_user_specified_data_fwf(full_pathlist[6], full_pathlist[7], full_pathlist[8], capsel, growsel)
        filtered_pointclouds, filtered_fwf_pointclouds, filtered_images = preprocessing.filter_data_for_selection_fwf(selected_pointclouds, selected_fwf_pointclouds, selected_images_augmented, elimper)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Resampling point clouds density based!" +'\n')
        output_log.configure(state="disabled")
        pointclouds_for_resampling = [preprocessing.load_point_cloud(file) for file in filtered_pointclouds]
        centered_points = preprocessing.center_point_cloud(pointclouds_for_resampling)
        resampled_pointclouds = np.array([preprocessing.resample_point_cloud_density_based(points, netpcsize) for points in centered_points])
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Generating metrics for point clouds!" +'\n')
        output_log.configure(state="disabled")
        combined_metrics_all = preprocessing.generate_metrics_for_selected_pointclouds_fwf_pred(filtered_pointclouds, filtered_fwf_pointclouds, full_pathlist[9], capsel, growsel, output_log)
        images_frontal, images_sideways = preprocessing.match_images_with_pointclouds(filtered_pointclouds, filtered_images)
        images_front = np.asarray(images_frontal)
        images_side = np.asarray(images_sideways)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Generating prediction data!" +'\n')
        output_log.configure(state="disabled")
        X_pc, X_metrics, X_img_1, X_img_2, onehot_to_label_dict = preprocessing.generate_prediction_data(capsel, growsel, filtered_pointclouds, resampled_pointclouds, combined_metrics_all, images_front, images_side, full_pathlist[9], 0.005)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Predicting for dataset!" +'\n')
        output_log.configure(state="disabled")
        model_utils.predict_for_data(pretrained_model, X_pc, X_metrics, X_img_1, X_img_2, onehot_to_label_dict, filtered_pointclouds, full_pathlist[1], model_dir, capsel, growsel, output_log)
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Extracting data!" +'\n')
        output_log.configure(state="disabled")
        full_pathlist = workspace_setup.extract_data_for_predictions(data_dir, work_dir, fwf_av, capsel, growsel, output_log)
        preprocessing.generate_colored_images(netimgsize, full_pathlist[6], full_pathlist[8])
        selected_pointclouds_augmented, selected_images_augmented = preprocessing.get_user_specified_data(full_pathlist[4], full_pathlist[5], capsel, growsel)
        filtered_pointclouds, filtered_images = preprocessing.filter_data_for_selection(selected_pointclouds_augmented, selected_images_augmented, elimper)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Resampling point clouds density based!" +'\n')
        output_log.configure(state="disabled")
        pointclouds_for_resampling = [preprocessing.load_point_cloud(file) for file in filtered_pointclouds]
        centered_points = preprocessing.center_point_cloud(pointclouds_for_resampling)
        resampled_pointclouds = np.array([preprocessing.resample_point_cloud_density_based(points, netpcsize) for points in centered_points])
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Generating metrics for point clouds!" +'\n')
        output_log.configure(state="disabled")
        combined_metrics_all = preprocessing.generate_metrics_for_selected_pointclouds_pred(filtered_pointclouds, full_pathlist[6], capsel, growsel, output_log)
        images_frontal, images_sideways = preprocessing.match_images_with_pointclouds(filtered_pointclouds, filtered_images)
        images_front = np.asarray(images_frontal)
        images_side = np.asarray(images_sideways)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Generating prediction data!" +'\n')
        output_log.configure(state="disabled")
        X_pc, X_metrics, X_img_1, X_img_2, onehot_to_label_dict = preprocessing.generate_prediction_data(capsel, growsel, filtered_pointclouds, resampled_pointclouds, combined_metrics_all, images_front, images_side, full_pathlist[6], 0.005)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - STATUS - Predicting for dataset!" +'\n')
        output_log.configure(state="disabled")
        model_utils.predict_for_data(pretrained_model, X_pc, X_metrics, X_img_1, X_img_2, onehot_to_label_dict, filtered_pointclouds, full_pathlist[1], model_dir, capsel, growsel, output_log)
        