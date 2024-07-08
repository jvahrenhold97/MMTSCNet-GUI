from threading import Thread
from enum import Enum, auto
import sys
import datetime
import os

class ReturnValueThread(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None
    def run(self):
        if self._target is None:
            return
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as exc:
            print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self.result

class TicketPurpose(Enum):
    UPDATE_TEXT = auto()

class Ticket:
    def __init__(self, ticket_type: TicketPurpose, ticket_value: str):
        self.ticket_type = ticket_type
        self.ticket_value = ticket_value

def validate_selected_folders(data_dir, work_dir, model_dir, start_btn, progbar, output_log):
    if data_dir == "Select Data Folder":
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        output_log.configure(state="normal")
        output_log.insert("1.0", f"{now_formatted} - ERROR - Please select a folder for your source data!" +'\n')
        output_log.configure(state="disabled")
        progbar.set(0)
        start_btn.configure(state="enabled", text="START")
        return False
    else:
        if work_dir == "Select Working Folder" or work_dir == data_dir:
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            output_log.configure(state="normal")
            output_log.insert("1.0", f"{now_formatted} - ERROR - Please select a working folder!" +'\n')
            output_log.configure(state="disabled")
            progbar.set(0)
            start_btn.configure(state="enabled", text="START")
            return False
        else:
            if model_dir == "Select Model Folder" or model_dir == work_dir or model_dir == data_dir:
                now = datetime.datetime.now()
                now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                output_log.configure(state="normal")
                output_log.insert("1.0", f"{now_formatted} - ERROR - Please select a folder for your trained models!" +'\n')
                output_log.configure(state="disabled")
                progbar.set(0)
                start_btn.configure(state="enabled", text="START")
                return False
            else:
                return True
            
def are_fwf_pointclouds_available(data_dir):
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