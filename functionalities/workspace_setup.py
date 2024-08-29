import os
import laspy as lp
import numpy as np
import zipfile
import shutil
from utils import main_utils
from scipy.spatial import KDTree
import datetime

def create_config_directory(local_pathlist, capsel, growsel, fwf_av, output_log):
    """
    Creates a temporary directory for MMTSCNet, dependent on the presence of FWF data.

    Args:
    local_pathlist: List of paths created for MMTSCNet.
    capsel: User-specified acquisition method.
    growsel: User-specified leaf condition.
    fwf_av: True/False - Presence of FWF data.

    Returns:
    local_pathlist: List of usable paths for further steps in the preprocessing
    """
    if fwf_av == True:
        config_dir = local_pathlist[4]
        local_dir = os.path.join(config_dir + "/DATA_" + capsel + "_" + growsel)
        local_las_dir = main_utils.join_paths(local_dir, "LAS")
        local_fwf_dir = main_utils.join_paths(local_dir, "FWF")
        local_img_dir = main_utils.join_paths(local_dir, "IMG")
        local_met_dir = main_utils.join_paths(local_dir, "MET")

        create_working_folder(local_dir, output_log)
        create_working_folder(local_las_dir, output_log)
        create_working_folder(local_fwf_dir, output_log)
        create_working_folder(local_img_dir, output_log)
        create_working_folder(local_met_dir, output_log)

        local_pathlist.append(local_dir)
        local_pathlist.append(local_las_dir)
        local_pathlist.append(local_fwf_dir)
        local_pathlist.append(local_img_dir)
        local_pathlist.append(local_met_dir)

        return local_pathlist
    else:
        config_dir = local_pathlist[2]
        local_dir = os.path.join(config_dir + "/DATA_" + capsel + "_" + growsel)
        local_las_dir = main_utils.join_paths(local_dir, "LAS")
        local_img_dir = main_utils.join_paths(local_dir, "IMG")
        local_met_dir = main_utils.join_paths(local_dir, "MET")

        create_working_folder(local_dir, output_log)
        create_working_folder(local_las_dir, output_log)
        create_working_folder(local_img_dir, output_log)
        create_working_folder(local_met_dir, output_log)

        local_pathlist.append(local_dir)
        local_pathlist.append(local_las_dir)
        local_pathlist.append(local_img_dir)
        local_pathlist.append(local_met_dir)
        return local_pathlist

def create_working_directory(workdir_path, fwf_av, output_log):
    """
    Creates a basic directory structure for MMTSCNet, dependent on the presence of FWF data.

    Args:
    workdir_path: User-specified working directory.
    fwf_av: True/False - Presence of FWF data.

    Returns:
    paths_to_create: List of usable paths for further steps in the preprocessing
    """
    unzipped_data_folder_name = "data_unzipped"
    configuration_data_folder_name = "data_config"
    fpc_name = "FPC"
    las_name = "LAS"
    fwf_name = "FWF"
    paths_to_create = []
    if fwf_av == True:
        source_data_unzipped_path = main_utils.join_paths(workdir_path, unzipped_data_folder_name)
        source_data_unzipped_path_las = main_utils.join_paths(source_data_unzipped_path, las_name)
        source_data_unzipped_path_fwf = main_utils.join_paths(source_data_unzipped_path, fwf_name)
        source_data_unzipped_path_fpc = main_utils.join_paths(source_data_unzipped_path, fpc_name)
        source_data_config_path = main_utils.join_paths(workdir_path, configuration_data_folder_name)
        paths_to_create.append(source_data_unzipped_path)
        paths_to_create.append(source_data_unzipped_path_las)
        paths_to_create.append(source_data_unzipped_path_fwf)
        paths_to_create.append(source_data_unzipped_path_fpc)
        paths_to_create.append(source_data_config_path)
    else:
        source_data_unzipped_path = main_utils.join_paths(workdir_path, unzipped_data_folder_name)
        source_data_unzipped_path_las = main_utils.join_paths(source_data_unzipped_path, las_name)
        source_data_config_path = main_utils.join_paths(workdir_path, configuration_data_folder_name)
        paths_to_create.append(source_data_unzipped_path)
        paths_to_create.append(source_data_unzipped_path_las)
        paths_to_create.append(source_data_config_path)
    for new_path in paths_to_create:
        create_working_folder(new_path, output_log)
    return paths_to_create

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
        output_log.insert("1.0", f"{now_formatted} - ERROR - Folder {path} already exists!" +'\n')
        output_log.configure(state="disabled")

def get_is_dataset_extracted(las_unzipped_path):
    """
    Check if dataset has been extracted already.

    Args:
    las_unzipped_path: Filepath to unzipped las files.

    Returns:
    True/False
    """
    extracted_datasets_count = 0
    for subdir in os.listdir(las_unzipped_path):
        extracted_datasets_count+=1
    if extracted_datasets_count > 0:
        return True
    else:
        return False
    
def get_las_and_fwf_base_dir_paths(data_source_path):
    """
    Retrieves filepaths to zipped las and FWF folders.

    Args:
    data_source_path: Filepath to source data.

    Returns:
    las_fwf_base_dir_paths: Paths to zipped las and fwf folders.
    """
    las_fwf_base_dir_paths = []
    for subdir in os.listdir(data_source_path):
        subdir_path = main_utils.join_paths(data_source_path, subdir)
        las_fwf_base_dir_paths.append(subdir_path)
    return las_fwf_base_dir_paths

def unzip_all_datasets(SOURCE_DATASET_PATH, pathlist, fwf_av, start_btn, progbar, output_log):
    """
    Unzips las and FWF data from the original dataset.

    Args:
    SOURCE_DATASET_PATH: Filepath to source data.
    pathlist: List of usable paths for MMTSCNet.
    fwf_av: True/False - Presence of FWF data.
    """
    if fwf_av == True:
        UNZIPPED_LAS_PATH = pathlist[1]
        UNZIPPED_FWF_PATH = pathlist[2]
        is_dataset_extracted = get_is_dataset_extracted(UNZIPPED_LAS_PATH)
        if is_dataset_extracted==False:
            base_data_folders = get_las_and_fwf_base_dir_paths(SOURCE_DATASET_PATH)
            for data_folder_path in base_data_folders:
                if data_folder_path.split("/")[1] == "las":
                    for file in os.listdir(data_folder_path):
                        now = datetime.datetime.now()
                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                        output_log.configure(state="normal")
                        output_log.insert("1.0", f"{now_formatted} - STATUS - Extracting files for plot {file}!" +'\n')
                        output_log.configure(state="disabled")
                        if "zip" in file:
                            FILE_PATH = main_utils.join_paths(data_folder_path, file)
                            dataset_name = file.split(".")[0]
                            DATASET_PATH = main_utils.join_paths(UNZIPPED_LAS_PATH, dataset_name)
                            os.mkdir(DATASET_PATH)
                            with zipfile.ZipFile(FILE_PATH, "r") as zip:
                                zip.extractall(DATASET_PATH)
                elif data_folder_path.split("/")[1] == "fwf":
                    for file in os.listdir(data_folder_path):
                        now = datetime.datetime.now()
                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                        output_log.configure(state="normal")
                        output_log.insert("1.0", f"{now_formatted} - STATUS - Extracting files for plot {file}!" +'\n')
                        output_log.configure(state="disabled")
                        if "zip" in file:
                            FILE_PATH = main_utils.join_paths(data_folder_path, file)
                            dataset_name = file.split(".")[0]
                            DATASET_PATH = main_utils.join_paths(UNZIPPED_FWF_PATH, dataset_name)
                            os.mkdir(DATASET_PATH)
                            with zipfile.ZipFile(FILE_PATH, "r") as zip:
                                zip.extractall(DATASET_PATH)
                else:
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    output_log.configure(state="normal")
                    output_log.insert("1.0", f"{now_formatted} - ERROR - Folders do not have the required structure, exiting!" +'\n')
                    output_log.configure(state="disabled")
                    progbar.set(0)
                    start_btn.configure(state="enabled", text="START")
        else:
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            output_log.configure(state="normal")
            output_log.insert("1.0", f"{now_formatted} - WARNING - Previously extracted dataset found, skipping!" +'\n')
            output_log.configure(state="disabled")
    else:
        UNZIPPED_LAS_PATH = pathlist[1]
        is_dataset_extracted = get_is_dataset_extracted(UNZIPPED_LAS_PATH)
        if is_dataset_extracted==False:
            base_data_folders = get_las_and_fwf_base_dir_paths(SOURCE_DATASET_PATH)
            for data_folder_path in base_data_folders:
                if data_folder_path.split("/")[1] == "las":
                    for file in os.listdir(data_folder_path):
                        now = datetime.datetime.now()
                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                        output_log.configure(state="normal")
                        output_log.insert("1.0", f"{now_formatted} - STATUS - Extracting files for plot {file}!" +'\n')
                        output_log.configure(state="disabled")
                        if "zip" in file:
                            FILE_PATH = main_utils.join_paths(data_folder_path, file)
                            dataset_name = file.split(".")[0]
                            DATASET_PATH = main_utils.join_paths(UNZIPPED_LAS_PATH, dataset_name)
                            os.mkdir(DATASET_PATH)
                            with zipfile.ZipFile(FILE_PATH, "r") as zip:
                                zip.extractall(DATASET_PATH)
                else:
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    output_log.configure(state="normal")
                    output_log.insert("1.0", f"{now_formatted} - ERROR - Folders do not have the required structure, exiting!" +'\n')
                    output_log.configure(state="disabled")
                    progbar.set(0)
                    start_btn.configure(state="enabled", text="START")
        else:
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            output_log.configure(state="normal")
            output_log.insert("1.0", f"{now_formatted} - WARNING - Previously extracted dataset found, skipping!" +'\n')
            output_log.configure(state="disabled")

def get_are_fwf_pcs_extracted(fwf_working_path):
    """
    Checks if the individual FWF flight strips have already been extracted.

    Args:
    fwf_working_path: Filepath to extracted FWF flight strips in the working directory.

    Returns:
    True/False
    """
    index=0
    for file in os.listdir(fwf_working_path):
        index+=1
    if index > 0:
        return True
    else: 
        return False

def getDimensions(file):
    """
    Retrieves the dimensions of a las file.

    Args:
    file: Las file.

    Returns:
    dimensions: Dimensions of the specified las file.
    """
    dimensions = ""
    for dim in file.point_format:
        dimensions += " " + dim.name
    return dimensions

def readLas(file):
    """
    Reads a las file and returns its points and dimensions.

    Args:
    file: Las file.

    Returns:
    source_cloud: Points included in the las file.
    dimensions: Dimensions of the specified las file.
    """
    dimensions = getDimensions(file)
    source_cloud = np.array([file.x, file.y, file.z]).T
    return source_cloud, dimensions

def append_to_las(in_laz, out_las, output_log):
    """
    Attaches one las file to another las file with the VLRs included.

    Args:
    in_laz: Filepath to the las file to attach.
    out_las: Filepath to the target las file.
    """
    with lp.open(out_las, mode='a') as outlas:
        with lp.open(in_laz) as inlas:
            # Copy VLRs if they are not already present in the output LAS
            if contains_full_waveform_data(in_laz, output_log) and not contains_full_waveform_data(out_las, output_log):
                for vlr in inlas.header.vlrs:
                    outlas.header.vlrs.append(vlr)
                for evlr in inlas.header.evlrs:
                    outlas.header.evlrs.append(evlr)
            for points in inlas.chunk_iterator(2_000_000):
                outlas.append_points(points)

def contains_full_waveform_data(las_file_path, output_log):
    """
    Checks a file for the presence of FWF data.

    Args:
    las_file_path: File to check for FWF data.

    Returns:
    True/False
    """
    try:
        las = lp.read(las_file_path)
        for vlr in las.header.vlrs:
            if 99 < vlr.record_id < 355:
                return True
        return False
    except Exception as e:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - ERROR - Error reading LAS file, {e}!" +'\n')
        output_log.configure(state="disabled")
        return False

def create_fpcs(fwf_unzipped_path, fpc_unzipped_path, output_log):
    """
    Attaches FWF flight strips to create a single FWF point cloud for each plot.

    Args:
    fwf_unzipped_path: Filepath to individual FWF flight strip point clouds.
    fpc_unzipped_path: Filepath to plot FWF point clouds.

    Returns:
    True/False
    """
    if get_are_fwf_pcs_extracted(fpc_unzipped_path) == False:
        for plot_folder in os.listdir(fwf_unzipped_path):
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            output_log.configure(state="normal")
            output_log.insert("1.0", f"{now_formatted} - STATUS - Creating FPC for plot {plot_folder}!" +'\n')
            output_log.configure(state="disabled")
            plot_path = main_utils.join_paths(fwf_unzipped_path, plot_folder)
            index = 0
            for fwf_file in os.listdir(plot_path):
                if fwf_file.lower().endswith(".las"):
                    if index == 0:
                        fwf_file_path = main_utils.join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = main_utils.join_paths(fpc_unzipped_path, savename)
                        shutil.copy2(fwf_file_path, out_las)
                        index+=1
                    else:
                        in_las = main_utils.join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = main_utils.join_paths(fpc_unzipped_path, savename)
                        append_to_las(in_las, out_las, output_log)
                        index+=1
                elif fwf_file.lower().endswith(".laz"):
                    if index == 0:
                        fwf_file_path = main_utils.join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = main_utils.join_paths(fpc_unzipped_path, savename)
                        shutil.copy2(fwf_file_path, out_las)
                        index+=1
                    else:
                        in_las = main_utils.join_paths(plot_path, fwf_file)
                        savename = str(plot_folder) + ".las"
                        out_las = main_utils.join_paths(fpc_unzipped_path, savename)
                        append_to_las(in_las, out_las, output_log)
                        index+=1
                else:
                    pass
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - WARNING - FPCs have previously been created, skipping!" +'\n')
        output_log.configure(state="disabled")

def get_capgrow(capsel, growsel):
    """
    Utility for validating acquisition and leaf-condition combinations.

    Args:
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.

    Returns:
    cap1: Acquisition method 1.
    cap2: Acquisition method 2.
    cap3: Acquisition method 3.
    grow1: Leaf-condition 1.
    grow2: Leaf-condition 2.
    """
    if capsel == "ALL":
        cap1 = "ALS"
        cap2 = "TLS"
        cap3 = "ULS"
        if growsel == "LEAF-ON":
            grow1 = "on"
            grow2 = "on"
            return cap1, cap2, cap3, grow1, grow2
        elif growsel == "LEAF-OFF":
            grow1 = "off"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
        else:
            grow1 = "on"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
    elif capsel == "ALS":
        cap1 = "ALS"
        cap2 = "ALS"
        cap3 = "ALS"
        if growsel == "LEAF-ON":
            grow1 = "on"
            grow2 = "on"
            return cap1, cap2, cap3, grow1, grow2
        elif growsel == "LEAF-OFF":
            grow1 = "off"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
        else:
            grow1 = "on"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
    elif capsel == "ULS":
        cap1 = "ULS"
        cap2 = "ULS"
        cap3 = "ULS"
        if growsel == "LEAF-ON":
            grow1 = "on"
            grow2 = "on"
            return cap1, cap2, cap3, grow1, grow2
        elif growsel == "LEAF-OFF":
            grow1 = "off"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
        else:
            grow1 = "on"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
    else:
        cap1 = "TLS"
        cap2 = "TLS"
        cap3 = "TLS"
        if growsel == "LEAF-ON":
            grow1 = "on"
            grow2 = "on"
            return cap1, cap2, cap3, grow1, grow2
        elif growsel == "LEAF-OFF":
            grow1 = "off"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2
        else:
            grow1 = "on"
            grow2 = "off"
            return cap1, cap2, cap3, grow1, grow2

def extract_single_trees_from_fpc(fpc_unzipped_path, las_unzipped_path, las_working_path, fwf_working_path, capsel, growsel, output_log):
    """
    Extraction of individual trees from FWF plot point clouds.

    Args:
    fpc_unzipped_path: Filepath to FWF plot point clouds.
    las_unzipped_path: Filepath to extracted las point clouds.
    las_working_path: Filepath to las point cloud target directory.
    fwf_working_path: Filepath to fwf point cloud target directory.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.
    """
    cap1, cap2, cap3, grow1, grow2 = get_capgrow(capsel, growsel)
    if get_are_fwf_pcs_extracted(fwf_working_path) == False:
        id_counter = 0
        tree_index = -1
        for fpc in os.listdir(fpc_unzipped_path):
            if fpc.lower().endswith(".las") or fpc.lower().endswith(".laz"):
                fpc_file_path = main_utils.join_paths(fpc_unzipped_path, fpc)
                fpc_name = fpc.split(".")[0]
                inFile = lp.read(fpc_file_path)
                fpc_source_cloud, fpc_header_text = readLas(inFile)
                kd_tree = KDTree(fpc_source_cloud[:, :3], leafsize=64)
                for plot in os.listdir(las_unzipped_path):
                    if plot == fpc_name:
                        plot_path = main_utils.join_paths(las_unzipped_path, plot)
                        for plot_pc_folder in os.listdir(plot_path):
                            if plot_pc_folder == "single_trees":
                                single_trees_plot_pc_folder = main_utils.join_paths(plot_path, plot_pc_folder)
                                for single_tree_pc_folder in os.listdir(single_trees_plot_pc_folder):
                                    single_tree_pc_folder_path = main_utils.join_paths(single_trees_plot_pc_folder, single_tree_pc_folder)
                                    tree_index+=1
                                    for single_tree_pc in os.listdir(single_tree_pc_folder_path):
                                        if single_tree_pc.lower().endswith(".laz") or single_tree_pc.lower().endswith(".las"):
                                            if cap1 in single_tree_pc or cap2 in single_tree_pc or cap3 in single_tree_pc:
                                                if grow1 in single_tree_pc or grow2 in single_tree_pc:
                                                    single_tree_pc_path = main_utils.join_paths(single_tree_pc_folder_path, single_tree_pc)
                                                    inFile_target = lp.read(single_tree_pc_path)
                                                    target_cloud, header_txt_target = readLas(inFile_target)
                                                    n_source = fpc_source_cloud.shape[0]
                                                    dist, idx = kd_tree.query(target_cloud[:, :3], k=10, eps=0.0)
                                                    idx = np.unique(idx)
                                                    idx = idx[idx != n_source]
                                                    exported_points = inFile.points[idx].copy()
                                                    outFile = lp.LasData(inFile.header)
                                                    outFile.vlrs = inFile.vlrs
                                                    outFile.points = exported_points
                                                    species = single_tree_pc.split("_")[0]
                                                    retrieval = single_tree_pc.split("_")[3]
                                                    method = single_tree_pc.split("_")[5].split(".")[0].split("-")[0]
                                                    if "on" in single_tree_pc:
                                                        output_path_fwf_pc = os.path.join(fwf_working_path + "/" + str(tree_index) + "_FWF_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_LEAF-ON_aug00.laz")
                                                        output_path_las_pc = os.path.join(las_working_path + "/" + str(tree_index) + "_REG_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_LEAF-ON_aug00.laz")
                                                        outFile.write(output_path_fwf_pc)
                                                        shutil.copy2(single_tree_pc_path, output_path_las_pc)
                                                        now = datetime.datetime.now()
                                                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                                                        output_log.configure(state="normal")
                                                        output_log.insert("1.0", f"{now_formatted} - STATUS - Extracted tree {id_counter} for plot {plot}!" +'\n')
                                                        output_log.configure(state="disabled")
                                                        id_counter+=1
                                                    else:
                                                        output_path_fwf_pc = os.path.join(fwf_working_path + "/" + str(tree_index) + "_FWF_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_LEAF-OFF_aug00.laz")
                                                        output_path_las_pc = os.path.join(las_working_path + "/" + str(tree_index) + "_REG_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_LEAF-OFF_aug00.laz")
                                                        outFile.write(output_path_fwf_pc)
                                                        shutil.copy2(single_tree_pc_path, output_path_las_pc)
                                                        now = datetime.datetime.now()
                                                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                                                        output_log.configure(state="normal")
                                                        output_log.insert("1.0", f"{now_formatted} - STATUS - Extracted tree {id_counter} for plot {plot}!" +'\n')
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
                pass
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - WARNING - FWF single trees have previously been extracted, skipping!" +'\n')
        output_log.configure(state="disabled")

def extract_single_trees_from_fpc_for_predictions(fpc_unzipped_path, las_unzipped_path, las_working_path, fwf_working_path, capsel, growsel, output_log):
    """
    Extraction of individual trees from FWF plot point clouds for predicting.

    Args:
    fpc_unzipped_path: Filepath to FWF plot point clouds.
    las_unzipped_path: Filepath to extracted las point clouds.
    las_working_path: Filepath to las point cloud target directory.
    fwf_working_path: Filepath to fwf point cloud target directory.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.
    """
    cap1, cap2, cap3, grow1, grow2 = get_capgrow(capsel, growsel)
    if get_are_fwf_pcs_extracted(fwf_working_path) == False:
        id_counter = 0
        tree_index = -1
        for fpc in os.listdir(fpc_unzipped_path):
            if fpc.lower().endswith(".las") or fpc.lower().endswith(".laz"):
                fpc_file_path = main_utils.join_paths(fpc_unzipped_path, fpc)
                fpc_name = fpc.split(".")[0]
                inFile = lp.read(fpc_file_path)
                fpc_source_cloud, fpc_header_text = readLas(inFile)
                kd_tree = KDTree(fpc_source_cloud[:, :3], leafsize=64)
                for plot in os.listdir(las_unzipped_path):
                    if plot == fpc_name:
                        plot_path = main_utils.join_paths(las_unzipped_path, plot)
                        for plot_pc_folder in os.listdir(plot_path):
                            if plot_pc_folder == "single_trees":
                                single_trees_plot_pc_folder = main_utils.join_paths(plot_path, plot_pc_folder)
                                for single_tree_pc_folder in os.listdir(single_trees_plot_pc_folder):
                                    single_tree_pc_folder_path = main_utils.join_paths(single_trees_plot_pc_folder, single_tree_pc_folder)
                                    tree_index+=1
                                    for single_tree_pc in os.listdir(single_tree_pc_folder_path):
                                        if single_tree_pc.lower().endswith(".laz") or single_tree_pc.lower().endswith(".las"):
                                            if cap1 in single_tree_pc or cap2 in single_tree_pc or cap3 in single_tree_pc:
                                                if grow1 in single_tree_pc or grow2 in single_tree_pc:
                                                    single_tree_pc_path = main_utils.join_paths(single_tree_pc_folder_path, single_tree_pc)
                                                    inFile_target = lp.read(single_tree_pc_path)
                                                    target_cloud, header_txt_target = readLas(inFile_target)
                                                    n_source = fpc_source_cloud.shape[0]
                                                    dist, idx = kd_tree.query(target_cloud[:, :3], k=10, eps=0.0)
                                                    idx = np.unique(idx)
                                                    idx = idx[idx != n_source]
                                                    exported_points = inFile.points[idx].copy()
                                                    outFile = lp.LasData(inFile.header)
                                                    outFile.vlrs = inFile.vlrs
                                                    outFile.points = exported_points
                                                    species = single_tree_pc.split("_")[0]
                                                    retrieval = single_tree_pc.split("_")[3]
                                                    method = single_tree_pc.split("_")[5].split(".")[0].split("-")[0]
                                                    if "on" in single_tree_pc:
                                                        output_path_fwf_pc = os.path.join(fwf_working_path + "/" + str(tree_index) + "_FWF_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_" + growsel + "_" + plot + ".laz")
                                                        output_path_las_pc = os.path.join(las_working_path + "/" + str(tree_index) + "_REG_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_" + growsel + "_" + plot + ".laz")
                                                        outFile.write(output_path_fwf_pc)
                                                        shutil.copy2(single_tree_pc_path, output_path_las_pc)
                                                        now = datetime.datetime.now()
                                                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                                                        output_log.configure(state="normal")
                                                        output_log.insert("1.0", f"{now_formatted} - STATUS - Extracted tree {id_counter} for plot {plot}!" +'\n')
                                                        output_log.configure(state="disabled")
                                                        id_counter+=1
                                                    else:
                                                        output_path_fwf_pc = os.path.join(fwf_working_path + "/" + str(tree_index) + "_FWF_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_" + growsel + "_" + plot + ".laz")
                                                        output_path_las_pc = os.path.join(las_working_path + "/" + str(tree_index) + "_REG_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_" + growsel + "_" + plot + ".laz")
                                                        outFile.write(output_path_fwf_pc)
                                                        shutil.copy2(single_tree_pc_path, output_path_las_pc)
                                                        now = datetime.datetime.now()
                                                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                                                        output_log.configure(state="normal")
                                                        output_log.insert("1.0", f"{now_formatted} - STATUS - Extracted tree {id_counter} for plot {plot}!" +'\n')
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
                pass
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - WARNING - FWF single trees have previously been extracted, skipping!" +'\n')
        output_log.configure(state="disabled")

def create_config_directory_for_predictions(local_pathlist, capsel, growsel, fwf_av, output_log):
    """
    Creates a temporary directory for MMTSCNet, dependent on the presence of FWF data.

    Args:
    local_pathlist: List of paths created for MMTSCNet.
    capsel: User-specified acquisition method.
    growsel: User-specified leaf condition.
    fwf_av: True/False - Presence of FWF data.

    Returns:
    local_pathlist: List of usable paths for further steps in the preprocessing
    """
    if fwf_av == True:
        config_dir = local_pathlist[4]
        local_dir = os.path.join(config_dir + "/PREDICTIONS_" + capsel + "_" + growsel)
        local_las_dir = main_utils.join_paths(local_dir, "LAS")
        local_fwf_dir = main_utils.join_paths(local_dir, "FWF")
        local_img_dir = main_utils.join_paths(local_dir, "IMG")
        local_met_dir = main_utils.join_paths(local_dir, "MET")

        create_working_folder(local_dir, output_log)
        create_working_folder(local_las_dir, output_log)
        create_working_folder(local_fwf_dir, output_log)
        create_working_folder(local_img_dir, output_log)
        create_working_folder(local_met_dir, output_log)

        local_pathlist.append(local_dir)
        local_pathlist.append(local_las_dir)
        local_pathlist.append(local_fwf_dir)
        local_pathlist.append(local_img_dir)
        local_pathlist.append(local_met_dir)

        return local_pathlist
    else:
        config_dir = local_pathlist[2]
        local_dir = os.path.join(config_dir + "/PREDICTIONS_" + capsel + "_" + growsel)
        local_las_dir = main_utils.join_paths(local_dir, "LAS")
        local_img_dir = main_utils.join_paths(local_dir, "IMG")
        local_met_dir = main_utils.join_paths(local_dir, "MET")

        create_working_folder(local_dir, output_log)
        create_working_folder(local_las_dir, output_log)
        create_working_folder(local_img_dir, output_log)
        create_working_folder(local_met_dir, output_log)

        local_pathlist.append(local_dir)
        local_pathlist.append(local_las_dir)
        local_pathlist.append(local_img_dir)
        local_pathlist.append(local_met_dir)
        return local_pathlist
    
def copy_files_for_prediction(las_unzipped_path, las_working_path, capsel, growsel):
    """
    Copies las point clouds to the working directory for predictions.

    Args:
    las_unzipped_path: Filepath to extracted las point clouds.
    las_working_path: Filepath to target working directory for las point clouds.
    capsel: User-specified acquisition method.
    growsel: User-specified leaf condition.
    """
    tree_index = 0
    id_counter = 0
    for plot_folder in os.listdir(las_unzipped_path):
        plot_path = os.path.join(las_unzipped_path, plot_folder)
        plot_name = plot_folder
        for subfolder in os.listdir(plot_path):
            if subfolder == "single_trees":
                id_counter += 1
                subfolder_path = os.path.join(plot_path, subfolder)
                for file in os.listdir(subfolder_path):
                    if file.lower().endswith(".las") or file.lower().endswith(".laz"):
                        cap1, cap2, cap3, grow1, grow2 = get_capgrow(capsel, growsel)
                        if cap1 in file or cap2 in file or cap3 in file:
                            if grow1 in file or grow2 in file:
                                species = file.split("_")[0]
                                retrieval = file.split("_")[3]
                                method = file.split("_")[5].split("-")[0]
                                filepath = os.path.join(subfolder_path, file)
                                las_working_path_pc = os.path.join(las_working_path + "/" + str(tree_index) + "_REG_" + species + "_" + method + "_" + retrieval + "_" + str(id_counter) + "_" + growsel + "_" + str(plot_name) + ".laz")
                                shutil.copy2(filepath, las_working_path_pc)
                                tree_index += 1

def extract_data_for_predictions(data_dir, work_dir, fwf_av, capsel, growsel, start_btn, progbar, output_log):
    """
    Main utility function for preparing data for predictions.

    Args:
    data_dir: User-specified source data directory.
    work_dir: User-specified working directory.
    fwf_av: True/False - Presence of FWF data.
    capsel: User-specified acquisition selection.
    growsel: User-specified leaf-confition selection.
    """
    local_pathlist = create_working_directory(work_dir, fwf_av, output_log)
    unzip_all_datasets(data_dir, local_pathlist, fwf_av, start_btn, progbar, output_log)
    if fwf_av == True:
        create_fpcs(local_pathlist[2], local_pathlist[3], output_log)
        full_pathlist = create_config_directory_for_predictions(local_pathlist, capsel, growsel, fwf_av, output_log)
        extract_single_trees_from_fpc_for_predictions(local_pathlist[3], local_pathlist[1], full_pathlist[6], full_pathlist[7], capsel, growsel, output_log)
        return full_pathlist
    else:
        full_pathlist = create_config_directory_for_predictions(local_pathlist, capsel, growsel, fwf_av, output_log)
        copy_files_for_prediction(local_pathlist[1], full_pathlist[6], capsel, growsel)
        return full_pathlist
