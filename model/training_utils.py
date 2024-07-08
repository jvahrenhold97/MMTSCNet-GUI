import tensorflow as tf
import keras
from model import hp_tuning_utils
import os
import datetime

def check_if_model_is_created(modeldir):
    files_list =  []
    for file in os.listdir(modeldir):
        if "TRAINED" in file:
            files_list.append(file)
        else:
            pass
    if len(files_list)>0:
        return True
    else:
        return False
    
def check_if_tuned_model_is_created(modeldir):
    files_list =  []
    for file in os.listdir(modeldir):
        if "tuning" in file:
            files_list.append(file)
        else:
            pass
    if len(files_list)>0:
        return True
    else:
        return False

def get_tuned_model_folder(modeldir):
    most_recent_file = None
    most_recent_time = None
    for file in os.listdir(modeldir):
        if file.lower().endswith(".tf") or file.lower().endswith(".keras") or file.lower().endswith(".h5"):
            date = file.split("_")[1].split(".")[0]
            filetime = datetime.datetime.strptime(date, "%Y%m%d-%H%M%S")
            if most_recent_time is None or filetime > most_recent_time:
                most_recent_file = file
                most_recent_time = filetime
        elif "TRAINED" in file:
            pass
        else:
            date = file.split("_")[1]
            filetime = datetime.datetime.strptime(date, "%Y%m%d-%H%M%S")
            if most_recent_time is None or filetime > most_recent_time:
                most_recent_file = file
                most_recent_time = filetime
    most_recent_path = os.path.join(modeldir + "/" + most_recent_file)
    return most_recent_path

def get_trained_model_folder(modeldir):
    most_recent_file = None
    most_recent_time = None
    for file in os.listdir(modeldir):
        if file.lower().endswith(".tf") or file.lower().endswith(".keras") or file.lower().endswith(".h5"):
            pass
        elif "TRAINED" in file:
            date = file.split("_")[1].split(".")[0]
            filetime = datetime.datetime.strptime(date, "%Y%m%d-%H%M%S")
            if most_recent_time is None or filetime > most_recent_time:
                most_recent_file = file
                most_recent_time = filetime
        else:
            pass
    most_recent_path = os.path.join(modeldir + "/" + most_recent_file)
    return most_recent_path

def load_tuned_model_from_folder(model_path):
    custom_objects = {
        'HyperParameters': hp_tuning_utils.HyperParameters,
        'TNetLess': hp_tuning_utils.TNetLess,
        'PointViewGCN': hp_tuning_utils.PointViewGCN,
        'EnhancedMetricsModel': hp_tuning_utils.EnhancedMetricsModel,
        'DenseNetModel': hp_tuning_utils.DenseNetModel,
        'CombinedModel': hp_tuning_utils.CombinedModel
    }
    model = keras.models.load_model(model_path, custom_objects)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7.5e-6),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="pr_curve", curve="PR"), tf.keras.metrics.PrecisionAtRecall(0.85, name="pr_at_rec"), tf.keras.metrics.RecallAtPrecision(0.85, name="rec_at_pr")]
    )
    return model

def load_trained_model_from_folder(model_path):
    model = keras.models.load_model(model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7.5e-6),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="pr_curve", curve="PR"), tf.keras.metrics.PrecisionAtRecall(0.85, name="pr_at_rec"), tf.keras.metrics.RecallAtPrecision(0.85, name="rec_at_pr")]
    )
    return model