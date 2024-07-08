from preprocessing import preprocessing_utils
from workspace_setup import workspace_setup_utils
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import datetime
from model import hp_tuning_utils
from sklearn.metrics import f1_score
import shutil
import laspy as lp
from scipy.spatial import KDTree

def check_if_model_is_created(modeldir):
    files_list =  []
    for file in os.listdir(modeldir):
        if "MMPNET" in file and file.lower().endswith(".keras"):
            files_list.append(file)
        else:
            pass
    if len(files_list)>0:
        return True
    else:
        return False

def generate_prediction_data(filtered_pointclouds, resampled_pointclouds, combined_metrics, images_frontal, images_sideways, metrics_dir, rfe_threshold, output_log):
    tree_labels = np.array(preprocessing_utils.get_labels_for_trees(filtered_pointclouds))
    label_encoder = LabelEncoder()
    elimination_labels = label_encoder.fit_transform(tree_labels)
    numeric_tree_labels = elimination_labels.astype(int)
    onehot_to_label_dict = {numeric_tree_labels[i]: tree_labels[i] for i in range(len(tree_labels))}
    rfe_metrics = []
    for file in os.listdir(metrics_dir):
        if "rfe" in file:
            rfe_metrics.append(file)
        else:
            pass
    if len(rfe_metrics) > 0:
        rfe_metrics_path = workspace_setup_utils.join_paths(metrics_dir, rfe_metrics[0])
        combined_eliminated_metrics = preprocessing_utils.load_metrics_from_path(rfe_metrics_path)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - INFO - Loaded metrics of shape {combined_eliminated_metrics.shape}!" +'\n')
        output_log.configure(state="disabled")
    else:
        combined_eliminated_metrics = preprocessing_utils.perform_recursive_feature_elimination_with_threshold(combined_metrics, elimination_labels, metrics_dir, output_log, rfe_threshold)
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - INFO - Metrics shape after Recursive Feature Elimination: {combined_eliminated_metrics.shape}!" +'\n')
        output_log.configure(state="disabled")
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - INFO - Tree species to predict for: {np.unique(tree_labels)}" +'\n')
    output_log.configure(state="disabled")
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"{now_formatted} - INFO - One-Hot-Encoding Labels!" +'\n')
    output_log.configure(state="disabled")
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(tree_labels.reshape(-1, 1))
    X_pc = resampled_pointclouds
    X_metrics = combined_eliminated_metrics
    X_img_1 = images_frontal
    X_img_2 = images_sideways
    return X_pc, X_metrics, X_img_1, X_img_2, y, onehot_to_label_dict

def predict_for_dataset(model, X_pc_pred, X_image_f_pred, X_image_s_pred, X_metrics_pred):
    # Ensure all inputs are numpy arrays
    X_pc_pred = np.array(X_pc_pred)
    X_image_f_pred = np.array(X_image_f_pred)
    X_image_s_pred = np.array(X_image_s_pred)
    X_metrics_pred = np.array(X_metrics_pred)

    inputs = []
    inputs.append(X_pc_pred)
    inputs.append(X_image_f_pred)
    inputs.append(X_image_s_pred)
    inputs.append(X_metrics_pred)

    print(X_pc_pred.shape)
    print(X_image_f_pred.shape)
    print(X_image_s_pred.shape)
    print(X_metrics_pred.shape)

    # Create a data generator for the prediction data
    batch_size = len(X_pc_pred)  # You can adjust the batch size as needed
    pred_gen = hp_tuning_utils.DataGenerator(X_pc_pred, X_image_f_pred, X_image_s_pred, X_metrics_pred, batch_size)
    
    # Predict using the model
    predictions = model.predict(pred_gen, verbose=1)
    predicted_classes = predictions.argmax(axis=-1)
    
    return predicted_classes

def evaluate_for_dataset(model, X_pc_pred, X_image_f_pred, X_image_s_pred, X_metrics_pred, y_true, predicted_classes, output_log):
    X_pc_pred, X_image_f_pred, X_image_s_pred, X_metrics_pred = hp_tuning_utils.normalize_data(X_pc_pred, X_image_f_pred, X_image_s_pred, X_metrics_pred)
    hp_tuning_utils.check_data(X_pc_pred, X_image_f_pred, X_image_s_pred, X_metrics_pred, y_true)
    corruption_found = hp_tuning_utils.check_label_corruption(y_true)
    if not corruption_found:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - INFO - No corruption found in one-hot encoded labels!" +'\n')
        output_log.configure(state="disabled")
    pred_gen = hp_tuning_utils.DataGenerator(X_pc_pred, X_image_f_pred, X_image_s_pred, X_metrics_pred, y_true, len(X_pc_pred))
    evaluation_results = model.evaluate(pred_gen, verbose=1)
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    output_log.configure(state="normal")
    output_log.insert("1.0", f"================================" +'\n')
    output_log.insert("1.0", f"F1-Score: {f1_score(y_true, predicted_classes, average='macro')}" +'\n')
    output_log.insert("1.0", f"Recall at Precision 0.85: {evaluation_results[6]}" +'\n')
    output_log.insert("1.0", f"Precision at Recall 0.85: {evaluation_results[5]}" +'\n')
    output_log.insert("1.0", f"AUCPR: {evaluation_results[4]}" +'\n')
    output_log.insert("1.0", f"Recall: {evaluation_results[3]}" +'\n')
    output_log.insert("1.0", f"Precision: {evaluation_results[2]}" +'\n')
    output_log.insert("1.0", f"Accuracy: {evaluation_results[1]}" +'\n')
    output_log.insert("1.0", f"Loss: {evaluation_results[0]}" +'\n')
    output_log.insert("1.0", f"=== MODEL EVALUATION RESULTS ===" +'\n')
    output_log.insert("1.0", f"================================" +'\n')
    output_log.configure(state="disabled")

def select_plot_pointclouds(copy_dest, editeable_pc_list, workspace_pathlist, fwf_av):
    unzipped_data_path, working_data_path, unzipped_las_path, unzipped_fwf_path, unzipped_fpc_path, working_las_path, working_fwf_path, working_img_path, metrics_path, useless = workspace_setup_utils.get_workspace_path_mappings(workspace_pathlist, fwf_av)
    plot_paths = []
    for path in editeable_pc_list:
        pc_name = path.split("/")[-1]
        capmet = pc_name.split("_")[3]
        leafcond = pc_name.split("_")[5]
        if leafcond == "LEAF-ON":
            lcond = "on"
        else:
            lcond = "off"
    for plot_dir in os.listdir(unzipped_las_path):
        for subfolder in os.listdir(plot_dir):
            if capmet in subfolder:
                for file in os.listdir(subfolder):
                    if capmet in file and lcond in file and file.lower().endswith(".laz"):
                        filepath = os.path.join(unzipped_las_path + "/" + plot_dir + "/" + subfolder + "/" + file)
                        savepath = workspace_setup_utils.join_paths(copy_dest, file)
                        shutil.copy2(filepath, savepath)
                        plot_paths.append(savepath)
    return plot_paths

def assert_numeric_labels(labels):
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    return numeric_labels

def classify_plot_pointcloud(plot_pcs, pcs, labels):
    for plot_pc in plot_pcs:
        ppc = lp.read(plot_pc)
        ppc_source_cloud, fpc_header_text = workspace_setup_utils.readLas(ppc)
        kd_tree = KDTree(ppc_source_cloud[:, :3], leafsize=64)
        for i in range(0, len(pcs)):
            pc = pcs[i]
            inFile_target = lp.read(pc)
            target_cloud, header_txt_target = workspace_setup_utils.readLas(inFile_target)
            n_source = ppc_source_cloud.shape[0]
            dist, idx = kd_tree.query(target_cloud[:, :3], k=10, eps=0.0)
            idx = np.unique(idx)
            idx = idx[idx != n_source]
            ppc.points[idx].classification = labels[i]
            with lp.open(plot_pc, mode='rw') as las_file:
                las_file.write_points(ppc.points)

def write_predictions_to_pointclouds(pc_list, label_dict, predicted_labels, workdir, workspace_pathlist):
    y_pred_real = hp_tuning_utils.map_onehot_to_real(predicted_labels, label_dict)
    editeable_pc_list = []
    editeable_pc_labels = []
    for i in range(0, len(pc_list)):
        pointcloud_path = pc_list[i]
        current_label = y_pred_real[i]
        pointcloud_name = pointcloud_path.split("/")[-1]
        if "aug00" in pointcloud_name:
            editeable_pc_list.append(pointcloud_path)
            editeable_pc_labels.append(current_label)
    dest_name = "data_predicted"
    copy_dest = workspace_setup_utils.join_paths(workdir, dest_name)
    workspace_setup_utils.create_working_folder(copy_dest)
    plot_pointclouds = select_plot_pointclouds(copy_dest, editeable_pc_list, workspace_pathlist)
    numeric_labels = assert_numeric_labels(editeable_pc_labels)
    classify_plot_pointcloud(plot_pointclouds, editeable_pc_list, numeric_labels)
    return copy_dest