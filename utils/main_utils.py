import os
import laspy as lp
import datetime

def validate_inputs(datadir, workdir, modeldir, elimper, maxpcscale, ssstest, capsel, growsel, batchsize, numpoints, start_btn, progbar, output_log):
    """
    Enables user-input argumnets to specify MMTSCNets functionalities.

    Args:
    datadir: See tooltips for args.
    workdir: See tooltips for args.
    modeldir: See tooltips for args.
    elimper: See tooltips for args.
    maxpcscale: See tooltips for args.
    ssstest: See tooltips for args.
    capsel: See tooltips for args.
    growsel: See tooltips for args.
    batchsize: See tooltips for args.
    numpoints: See tooltips for args.

    Returns:
    args: List of possible user-inputs.
    """
    data_dir = datadir
    work_dir = workdir
    model_dir = modeldir
    if elimper < 99 and elimper > 0:
        elim_per = elimper
    elif elimper == 0:
        elim_per = elimper
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - ERROR - Elimination percentage is not in a valid range (0 - 100)!" +'\n')
        output_log.configure(state="disabled")
        progbar.set(0)
        start_btn.configure(state="enabled", text="START")
    if maxpcscale > 0.001 and maxpcscale < 0.01:
        max_pcscale = maxpcscale
    elif maxpcscale < 0.001 or maxpcscale > 0.01:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - ERROR - Scaling factor is too lagre/small, exiting!" +'\n')
        output_log.configure(state="disabled")
        progbar.set(0)
        start_btn.configure(state="enabled", text="START")
    if ssstest > 0.05 and ssstest < 0.5:
        sss_test = ssstest
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - ERROR - Train-Test split ratio is too large/small, exiting!" +'\n')
        output_log.configure(state="disabled")
        progbar.set(0)
        start_btn.configure(state="enabled", text="START")
    if capsel == "ALS" or capsel == "ULS" or capsel == "TLS" or capsel == "ALL":
        cap_sel = capsel
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - ERROR - Capture selection can only be [ALS | TLS | ULS | ALL]!" +'\n')
        output_log.configure(state="disabled")
        progbar.set(0)
        start_btn.configure(state="enabled", text="START")
    if growsel == "LEAF-ON" or growsel == "LEAF-OFF" or growsel == "ALL":
        grow_sel = growsel
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - ERROR - Growth selection can only be [LEAF-ON | LEAF-OFF | ALL]!" +'\n')
        output_log.configure(state="disabled")
        progbar.set(0)
        start_btn.configure(state="enabled", text="START")
    if batchsize == 4 or batchsize == 8 or batchsize == 16 or batchsize == 32:
        bsize = batchsize
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - ERROR - Batch size can only be [4 | 8 | 16 | 32]!" +'\n')
        output_log.configure(state="disabled")
        progbar.set(0)
        start_btn.configure(state="enabled", text="START")
    if numpoints == 512 or numpoints == 1024 or numpoints == 2048 or numpoints == 4096:
        pc_size = numpoints
    else:
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - ERROR - Point cloud size can only be [512 | 1024 | 2048 | 4096]!" +'\n')
        output_log.configure(state="disabled")
        progbar.set(0)
        start_btn.configure(state="enabled", text="START")
    img_size = 224
    return data_dir, work_dir, model_dir, elim_per, max_pcscale, sss_test, cap_sel, grow_sel, bsize, img_size, pc_size

def are_fwf_pointclouds_available(data_dir):
    """
    Checks for the presence of FWF data.

    Args:
    data_dir: Directory containing source data.

    Returns:
    True/False
    """
    fwf_folders = []
    for subfolder in os.listdir(data_dir):
        if "fwf" in subfolder or "FWF" in subfolder:
            fwf_folders.append(subfolder)
        else:
            pass
    if len(fwf_folders) > 0:
        return True
    else:
        return False
    
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

def check_if_model_is_created(modeldir):
    """
    Checks for the presence of a previously trained instance of MMTSCNet.

    Args:
    modeldir: Directory containing models.

    Returns:
    True/False
    """
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
    
def copy_las_file_with_laspy(src, dest):
    """
    Copies a .las or .laz file with all existing data (including VLRs).

    Args:
    src: Source file path.
    dest: Destination file path.
    """
    # Read the source file
    with lp.open(src) as src_las:
        header = src_las.header
        points = src_las.points
        vlrs = src_las.header.vlrs
        evlrs = src_las.header.evlrs
    # Write to the destination file
    with lp.open(dest, mode='w', header=header) as dest_las:
        dest_las.points = points
        dest_las.header.vlrs.extend(vlrs)
        dest_las.header.evlrs.extend(evlrs)

def contains_full_waveform_data(las_file_path):
    """
    Checks for the presence of FWFdata in a .las file.

    Args:
    las_file_path: Path to the .las file.

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
        return False