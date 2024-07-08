import os
import datetime
import zipfile
import shutil
import laspy as lp
import numpy as np
from scipy.spatial import KDTree

def create_working_directory(workdir_path, output_log, fwf_av):
    WORKDIR_PATH = workdir_path
    
    unzipped_data_folder_name = "data_unzipped"
    working_data_folder_name = "data_working"
    configuration_data_folder_name = "data_config"
    
    fpc_name = "fpc"
    las_name = "las"
    fwf_name = "fwf"
    img_name = "img"
    metrics_name = "metrics"
    
    paths_to_create = []

    if fwf_av == True:
        SOURCE_DATA_UNZIPPED_PATH = join_paths(WORKDIR_PATH, unzipped_data_folder_name)
        SOURCE_DATA_WORKING_PATH = join_paths(WORKDIR_PATH, working_data_folder_name)
        UNZIPPED_SOURCE_DATA_PATH_LAS = join_paths(SOURCE_DATA_UNZIPPED_PATH, las_name)
        UNZIPPED_SOURCE_DATA_PATH_FWF = join_paths(SOURCE_DATA_UNZIPPED_PATH, fwf_name)
        UNZIPPED_SOURCE_DATA_PATH_FWF_FPC = join_paths(SOURCE_DATA_UNZIPPED_PATH, fpc_name)
        WORKING_SOURCE_DATA_PATH_LAS = join_paths(SOURCE_DATA_WORKING_PATH, las_name)
        WORKING_SOURCE_DATA_PATH_FWF = join_paths(SOURCE_DATA_WORKING_PATH, fwf_name)
        WORKING_SOURCE_DATA_PATH_IMG = join_paths(SOURCE_DATA_WORKING_PATH, img_name)
        WORKING_SOURCE_DATA_PATH_METRICS = join_paths(SOURCE_DATA_WORKING_PATH, metrics_name)
        CONFIGURATION_SOURCE_PATH = join_paths(WORKDIR_PATH, configuration_data_folder_name)
        
        paths_to_create.append(SOURCE_DATA_UNZIPPED_PATH)
        paths_to_create.append(SOURCE_DATA_WORKING_PATH)
        paths_to_create.append(UNZIPPED_SOURCE_DATA_PATH_LAS)
        paths_to_create.append(UNZIPPED_SOURCE_DATA_PATH_FWF)
        paths_to_create.append(UNZIPPED_SOURCE_DATA_PATH_FWF_FPC)
        paths_to_create.append(WORKING_SOURCE_DATA_PATH_LAS)
        paths_to_create.append(WORKING_SOURCE_DATA_PATH_FWF)
        paths_to_create.append(WORKING_SOURCE_DATA_PATH_IMG)
        paths_to_create.append(WORKING_SOURCE_DATA_PATH_METRICS)
        paths_to_create.append(CONFIGURATION_SOURCE_PATH)
    else:
        SOURCE_DATA_UNZIPPED_PATH = join_paths(WORKDIR_PATH, unzipped_data_folder_name)
        SOURCE_DATA_WORKING_PATH = join_paths(WORKDIR_PATH, working_data_folder_name)
        UNZIPPED_SOURCE_DATA_PATH_LAS = join_paths(SOURCE_DATA_UNZIPPED_PATH, las_name)
        WORKING_SOURCE_DATA_PATH_LAS = join_paths(SOURCE_DATA_WORKING_PATH, las_name)
        WORKING_SOURCE_DATA_PATH_IMG = join_paths(SOURCE_DATA_WORKING_PATH, img_name)
        WORKING_SOURCE_DATA_PATH_METRICS = join_paths(SOURCE_DATA_WORKING_PATH, metrics_name)
        CONFIGURATION_SOURCE_PATH = join_paths(WORKDIR_PATH, configuration_data_folder_name)
        
        paths_to_create.append(SOURCE_DATA_UNZIPPED_PATH)
        paths_to_create.append(SOURCE_DATA_WORKING_PATH)
        paths_to_create.append(UNZIPPED_SOURCE_DATA_PATH_LAS)
        paths_to_create.append(WORKING_SOURCE_DATA_PATH_LAS)
        paths_to_create.append(WORKING_SOURCE_DATA_PATH_IMG)
        paths_to_create.append(WORKING_SOURCE_DATA_PATH_METRICS)
        paths_to_create.append(CONFIGURATION_SOURCE_PATH)
    
    for new_path in paths_to_create:
        create_working_folder(new_path, output_log)
        
    return paths_to_create

def join_paths(path, folder_name):
    """
    Joins a filepath with a folder name to create a new filepath.

    Args:
    path: Filepath to directory where folder is supposed to be created.
    folder_name: Name of the folder to be created.

    Returns:
    full_path: The path with the appended folder name.
    """
    full_path = os.path.join(path + "/" + folder_name)
    return full_path

def create_working_folder(path, output_log):
    """
    Creates a folder at the input filepath.

    Args:
    path: Filepath with folder name appended for creation.
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - INFO - Folder {path} already exists!" +'\n')
        output_log.configure(state="disabled")

def get_is_dataset_extracted(las_unzipped_path):
    extracted_datasets_count = 0
    for subdir in os.listdir(las_unzipped_path):
        extracted_datasets_count+=1
    if extracted_datasets_count > 0:
        return True
    else:
        return False
    
def unzip_all_datasets(SOURCE_DATASET_PATH, pathlist, start_btn, progbar, output_log, fwf_av):
    if fwf_av == True:
        UNZIPPED_LAS_PATH = pathlist[2]
        UNZIPPED_FWF_PATH = pathlist[3]
        is_dataset_extracted = get_is_dataset_extracted(UNZIPPED_LAS_PATH)
        if is_dataset_extracted==False:
            base_data_folders = get_las_and_fwf_base_dir_paths(SOURCE_DATASET_PATH)
            for data_folder_path in base_data_folders:
                if data_folder_path.split("/")[1] == "las":
                    for file in os.listdir(data_folder_path):
                        now = datetime.datetime.now()
                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                        output_log.configure(state="normal")
                        output_log.insert("1.0", f"{now_formatted} - INFO - Extracting files for plot {file}!" +'\n')
                        output_log.configure(state="disabled")
                        if "zip" in file:
                            FILE_PATH = join_paths(data_folder_path, file)
                            dataset_name = file.split(".")[0]
                            DATASET_PATH = join_paths(UNZIPPED_LAS_PATH, dataset_name)
                            os.mkdir(DATASET_PATH)
                            with zipfile.ZipFile(FILE_PATH, "r") as zip:
                                zip.extractall(DATASET_PATH)
                elif data_folder_path.split("/")[1] == "fwf":
                    for file in os.listdir(data_folder_path):
                        now = datetime.datetime.now()
                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                        output_log.configure(state="normal")
                        output_log.insert("1.0", f"{now_formatted} - INFO - Extracting files for plot {file}!" +'\n')
                        output_log.configure(state="disabled")
                        if "zip" in file:
                            FILE_PATH = join_paths(data_folder_path, file)
                            dataset_name = file.split(".")[0]
                            DATASET_PATH = join_paths(UNZIPPED_FWF_PATH, dataset_name)
                            os.mkdir(DATASET_PATH)
                            with zipfile.ZipFile(FILE_PATH, "r") as zip:
                                zip.extractall(DATASET_PATH)
                else:
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    output_log.configure(state="normal")
                    output_log.insert("1.0", f"{now_formatted} - INFO - Folders don't have the correct structure, see Documentation for help, exiting!" +'\n')
                    output_log.configure(state="disabled")
                    start_btn.configure(state="normal")
                    progbar.set(0)
        else:
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            output_log.configure(state="normal")
            output_log.insert("1.0", f"{now_formatted} - INFO - Extracted dataset found, skipping!" +'\n')
            output_log.configure(state="disabled")
    else:
        UNZIPPED_LAS_PATH = pathlist[2]
        is_dataset_extracted = get_is_dataset_extracted(UNZIPPED_LAS_PATH)
        if is_dataset_extracted==False:
            base_data_folders = get_las_and_fwf_base_dir_paths(SOURCE_DATASET_PATH)
            for data_folder_path in base_data_folders:
                if data_folder_path.split("/")[1] == "las":
                    for file in os.listdir(data_folder_path):
                        now = datetime.datetime.now()
                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                        output_log.configure(state="normal")
                        output_log.insert("1.0", f"{now_formatted} - INFO - Extracting files for plot {file}!" +'\n')
                        output_log.configure(state="disabled")
                        if "zip" in file:
                            FILE_PATH = join_paths(data_folder_path, file)
                            dataset_name = file.split(".")[0]
                            DATASET_PATH = join_paths(UNZIPPED_LAS_PATH, dataset_name)
                            os.mkdir(DATASET_PATH)
                            with zipfile.ZipFile(FILE_PATH, "r") as zip:
                                zip.extractall(DATASET_PATH)
                else:
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    output_log.configure(state="normal")
                    output_log.insert("1.0", f"{now_formatted} - INFO - Folders don't have the correct structure, see Documentation for help, exiting!" +'\n')
                    output_log.configure(state="disabled")
                    start_btn.configure(state="normal")
                    progbar.set(0)
        else:
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            output_log.configure(state="normal")
            output_log.insert("1.0", f"{now_formatted} - INFO - Extracted dataset found, skipping!" +'\n')
            output_log.configure(state="disabled")

def get_las_and_fwf_base_dir_paths(data_source_path):
    las_fwf_base_dir_paths = []
    for subdir in os.listdir(data_source_path):
        subdir_path = join_paths(data_source_path, subdir)
        las_fwf_base_dir_paths.append(subdir_path)
    return las_fwf_base_dir_paths

def get_workspace_path_mappings(workspace_path_list, fwf_av):
    """
    Extracts a list of filepaths to recieve all included paths.

    Args:
    workspace_path_list: List of filepaths.

    Returns:
    data_unzipped_path: Path to the folder where source data should be unzipped.
    data_working_path: Path to the folder used for data processing.
    las_unzipped_path: Path where .las/.laz files will be extracted to.
    fwf_unzipped_path: Path where full waveform pointclouds will be extracted to.
    fpc_unzipped_path: Path where combined full waveform pointclouds will be saved.
    las_working_path: Path where .las/.laz files will be saved after preprocessing.
    fwf_working_path: Path where full waveform pointclouds will be saved after preprocessing.
    img_working_path: Path where generated images will be saved.
    metrics_path: Path where metrics generated metrics will be saved.
    """
    if fwf_av == True:
        data_unzipped_path = workspace_path_list[0]
        data_working_path = workspace_path_list[1]
        las_unzipped_path = workspace_path_list[2]
        fwf_unzipped_path = workspace_path_list[3]
        fpc_unzipped_path = workspace_path_list[4]
        las_working_path = workspace_path_list[5]
        fwf_working_path = workspace_path_list[6]
        img_working_path = workspace_path_list[7]
        metrics_path = workspace_path_list[8]
        config_data_path = workspace_path_list[9]
        return data_unzipped_path, data_working_path, las_unzipped_path, fwf_unzipped_path, fpc_unzipped_path, las_working_path, fwf_working_path, img_working_path, metrics_path, config_data_path
    else:
        data_unzipped_path = workspace_path_list[0]
        data_working_path = workspace_path_list[1]
        las_unzipped_path = workspace_path_list[2]
        las_working_path = workspace_path_list[3]
        img_working_path = workspace_path_list[4]
        metrics_path = workspace_path_list[5]
        config_data_path = workspace_path_list[6]
        return data_unzipped_path, data_working_path, las_unzipped_path, las_working_path, img_working_path, metrics_path, config_data_path

def get_are_fwf_pcs_extracted(fwf_working_path):
    index=0
    for file in os.listdir(fwf_working_path):
        index+=1
    if index > 0:
        return True
    else: 
        return False
    
def append_to_las(in_laz, out_las):
    with lp.open(out_las, mode='a') as outlas:
        with lp.open(in_laz) as inlas:
            for points in inlas.chunk_iterator(2_000_000):
                outlas.append_points(points)

def getDimensions(file):
    dimensions = ""
    for dim in file.point_format:
        dimensions += " " + dim.name
    return dimensions

def readLas(file):
    dimensions = getDimensions(file)
    source_cloud = np.array([file.x, file.y, file.z]).T
    return source_cloud, dimensions

def create_fpcs(fwf_unzipped_path, fpc_unzipped_path, output_log):
    if get_are_fwf_pcs_extracted(fpc_unzipped_path) == False:
        for plot_folder in os.listdir(fwf_unzipped_path):
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            output_log.configure(state="normal")
            output_log.insert("1.0", f"{now_formatted} - INFO - Creating full-waveform pointclouds for plot {plot_folder}!" +'\n')
            output_log.configure(state="disabled")
            plot_path = join_paths(fwf_unzipped_path, plot_folder)
            index = 0
            for fwf_file in os.listdir(plot_path):
                if fwf_file.lower().endswith(".las"):
                    if index == 0:
                        fwf_file_path = join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = join_paths(fpc_unzipped_path, savename)
                        shutil.copy2(fwf_file_path, out_las)
                        index+=1
                    else:
                        in_las = join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = join_paths(fpc_unzipped_path, savename)
                        append_to_las(in_las, out_las)
                        index+=1
                elif fwf_file.lower().endswith(".laz"):
                    if index == 0:
                        fwf_file_path = join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = join_paths(fpc_unzipped_path, savename)
                        shutil.copy2(fwf_file_path, out_las)
                        index+=1
                    else:
                        in_las = join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = join_paths(fpc_unzipped_path, savename)
                        append_to_las(in_las, out_las)
                        index+=1
                else:
                    pass
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - INFO - Full-waveform pointclouds have already been created!" +'\n')
        output_log.configure(state="disabled")

def extract_single_trees_from_fpc(fpc_unzipped_path, las_unzipped_path, las_working_path, fwf_working_path, output_log):
    if get_are_fwf_pcs_extracted(fwf_working_path) == False:
        id_counter = 0
        tree_index = -1
        for fpc in os.listdir(fpc_unzipped_path):
            if fpc.lower().endswith(".las") or fpc.lower().endswith(".laz"):
                fpc_file_path = join_paths(fpc_unzipped_path, fpc)
                fpc_name = fpc.split(".")[0]
                inFile = lp.read(fpc_file_path)
                fpc_source_cloud, fpc_header_text = readLas(inFile)
                kd_tree = KDTree(fpc_source_cloud[:, :3], leafsize=64)
                for plot in os.listdir(las_unzipped_path):
                    if plot == fpc_name:
                        plot_path = join_paths(las_unzipped_path, plot)
                        for plot_pc_folder in os.listdir(plot_path):
                            if plot_pc_folder == "single_trees":
                                single_trees_plot_pc_folder = join_paths(plot_path, plot_pc_folder)
                                for single_tree_pc_folder in os.listdir(single_trees_plot_pc_folder):
                                    single_tree_pc_folder_path = join_paths(single_trees_plot_pc_folder, single_tree_pc_folder)
                                    tree_index+=1
                                    for single_tree_pc in os.listdir(single_tree_pc_folder_path):
                                        if single_tree_pc.lower().endswith(".laz") or single_tree_pc.lower().endswith(".las"):
                                            single_tree_pc_path = join_paths(single_tree_pc_folder_path, single_tree_pc)
                                            inFile_target = lp.read(single_tree_pc_path)
                                            target_cloud, header_txt_target = readLas(inFile_target)
                                            n_source = fpc_source_cloud.shape[0]
                                            dist, idx = kd_tree.query(target_cloud[:, :3], k=10, eps=0.0)
                                            idx = np.unique(idx)
                                            idx = idx[idx != n_source]
                                            exported_points = inFile.points[idx].copy()
                                            species = single_tree_pc.split("_")[0]
                                            retrieval = single_tree_pc.split("_")[3]
                                            method = single_tree_pc.split("_")[5].split(".")[0].split("-")[0]
                                            if "on" in single_tree_pc:
                                                output_path_fwf_pc = os.path.join(fwf_working_path + "/" + str(tree_index) + "_FWF_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_LEAF-ON_aug00.laz")
                                                output_path_las_pc = os.path.join(las_working_path + "/" + str(tree_index) + "_REG_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_LEAF-ON_aug00.laz")
                                                outFile = lp.LasData(inFile.header)
                                                outFile.points = exported_points
                                                outFile.write(output_path_fwf_pc)
                                                shutil.copy2(single_tree_pc_path, output_path_las_pc)
                                                now = datetime.datetime.now()
                                                now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                                                output_log.configure(state="normal")
                                                output_log.insert("1.0", f"{now_formatted} - INFO - Extracted tree #{id_counter} for plot {plot}!" +'\n')
                                                output_log.configure(state="disabled")
                                                id_counter+=1
                                            else:
                                                output_path_fwf_pc = os.path.join(fwf_working_path + "/" + str(tree_index) + "_FWF_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_LEAF-OFF_aug00.laz")
                                                output_path_las_pc = os.path.join(las_working_path + "/" + str(tree_index) + "_REG_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_LEAF-OFF_aug00.laz")
                                                outFile = lp.LasData(inFile.header)
                                                outFile.points = exported_points
                                                outFile.write(output_path_fwf_pc)
                                                shutil.copy2(single_tree_pc_path, output_path_las_pc)
                                                now = datetime.datetime.now()
                                                now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                                                output_log.configure(state="normal")
                                                output_log.insert("1.0", f"{now_formatted} - INFO - Extracted tree #{id_counter} for plot {plot}!" +'\n')
                                                output_log.configure(state="disabled")
                                                id_counter+=1
                                        else:
                                            pass
                            else:
                                pass
                    else:
                        pass
            else:
                pass
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - INFO - Full-waveform single trees have already been extracted!" +'\n')
        output_log.configure(state="disabled")