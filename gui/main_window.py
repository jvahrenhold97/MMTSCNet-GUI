import customtkinter
from PIL import Image
import os
from customtkinter import *
import tkinter as tk
import datetime
from queue import Queue
from gui import gui_utils, main_functions
from model import training_utils
import webbrowser

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("dark-blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        heading_font = customtkinter.CTkFont("Calibri", 20)
        text_font = customtkinter.CTkFont("Calibri", 14)

        elimpercentage = tk.IntVar(self, 0)
        ssstestsize = tk.IntVar(self, 20)
        train_var = tk.StringVar(value="on")
        tune_var = tk.StringVar(value="on")

        cw = os.getcwd()

        self.queue_message = Queue()
        self.bind("<<CheckQueue>>", self.check_queue)

        self.title("MMPNet-UI")
        self.geometry("1340x924")
        self.iconbitmap(os.path.join(cw, "images/logo_light.png"))

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0, fg_color=("gray86", "gray17"))
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        light_image_path = os.path.join(cw, "images/logo_light.png")
        dark_image_path = os.path.join(cw, "images/logo_dark.png")
        my_image = customtkinter.CTkImage(light_image=Image.open(light_image_path),
                                  dark_image=Image.open(dark_image_path),
                                  size=(140, 140))
        self.logo_button = customtkinter.CTkButton(self, image=my_image, text="", fg_color="transparent", bg_color=("gray86", "gray17"), command=self.open_website)
        self.logo_button.grid(row=0, column=0, padx=10, pady=(10, 10), sticky="n")
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w", font=text_font)
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], font=text_font, command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w", font=text_font)
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%"], font=text_font, command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        self.options_frame = customtkinter.CTkFrame(self, fg_color=("gray86", "gray17"))
        self.options_frame.grid(row=0, column=1, rowspan=8, padx=(10, 10), pady=(10, 10), sticky="nsew")
        self.options_frame.grid_columnconfigure(0, weight=1)

        self.mode_label = customtkinter.CTkLabel(self.options_frame, text="MMPNET-SETUP", fg_color="transparent", font=heading_font)
        self.mode_label.grid(row=0, column=0, padx=10, pady=10)

        self.options_frame_interact = customtkinter.CTkFrame(self.options_frame, fg_color="transparent")
        self.options_frame_interact.grid(row=1, column=0, rowspan=7, padx=(10, 10), pady=(10, 10), sticky="nsew")
        self.options_frame_interact.grid_columnconfigure((0, 2), weight=2)
        self.options_frame_interact.grid_columnconfigure(1, weight=1)
        self.options_frame_interact.grid_rowconfigure((0,1,2,3,4,5,6,7,8,9,10), weight=1)

        self.input_label_01 = customtkinter.CTkLabel(self.options_frame_interact, text="Data-Directory:", fg_color="transparent", font=text_font)
        self.input_label_01.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.folder_select_button_01 = customtkinter.CTkButton(self.options_frame_interact, text="Select Data Folder", font=text_font, command=self.file_select_data)
        self.folder_select_button_01.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        self.input_label_02 = customtkinter.CTkLabel(self.options_frame_interact, text="Working-Directory:", fg_color="transparent", font=text_font)
        self.input_label_02.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.folder_select_button_02 = customtkinter.CTkButton(self.options_frame_interact, text="Select Working Folder", font=text_font, command=self.file_select_work)
        self.folder_select_button_02.grid(row=1, column=2, padx=10, pady=10, sticky="ew")

        self.input_label_03 = customtkinter.CTkLabel(self.options_frame_interact, text="Model-Directory:", fg_color="transparent", font=text_font)
        self.input_label_03.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.folder_select_button_03 = customtkinter.CTkButton(self.options_frame_interact, text="Select Model Folder", font=text_font, command=self.file_select_model)
        self.folder_select_button_03.grid(row=2, column=2, padx=10, pady=10, sticky="ew")

        self.elimper_label = customtkinter.CTkLabel(self.options_frame_interact, text="Elimination percentage:", fg_color="transparent", font=text_font)
        self.elimper_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.elimper_label_num = customtkinter.CTkLabel(self.options_frame_interact, textvariable=elimpercentage, fg_color="transparent", font=text_font)
        self.elimper_label_num.grid(row=3, column=1, pady=10, sticky="e")
        self.slider_elimper = customtkinter.CTkSlider(self.options_frame_interact, from_=0, to=50, number_of_steps=50, variable=elimpercentage)
        self.slider_elimper.grid(row=3, column=2, padx=(10, 10), pady=(10, 10), sticky="ew")

        self.ssstest_label = customtkinter.CTkLabel(self.options_frame_interact, text="Stratified Shuffle-Split Validation percentage:", fg_color="transparent", font=text_font)
        self.ssstest_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.ssstest_label_num = customtkinter.CTkLabel(self.options_frame_interact, textvariable=ssstestsize, fg_color="transparent", font=text_font)
        self.ssstest_label_num.grid(row=4, column=1, pady=10, sticky="e")
        self.slider_ssstest = customtkinter.CTkSlider(self.options_frame_interact, from_=20, to=40, number_of_steps=20, variable=ssstestsize)
        self.slider_ssstest.grid(row=4, column=2, padx=(10, 10), pady=(10, 10), sticky="ew")

        self.capsel_label = customtkinter.CTkLabel(self.options_frame_interact, text="Capture method selection:", fg_color="transparent", font=text_font)
        self.capsel_label.grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.capsel_button = customtkinter.CTkSegmentedButton(self.options_frame_interact, values=["ULS", "ALS", "TLS", "ALL"], command=self.capsel_button_callback, font=text_font)
        self.capsel_button.set("ULS")
        self.capsel_button.grid(row=5, column=2, padx=(10, 10), pady=(10, 10), sticky="ew")

        self.growsel_label = customtkinter.CTkLabel(self.options_frame_interact, text="Leaf condition selection:", fg_color="transparent", font=text_font)
        self.growsel_label.grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.growsel_button = customtkinter.CTkSegmentedButton(self.options_frame_interact, values=["LEAF-ON", "LEAF-OFF", "ALL"], command=self.growsel_button_callback, font=text_font)
        self.growsel_button.set("LEAF-ON")
        self.growsel_button.grid(row=6, column=2, padx=(10, 10), pady=(10, 10), sticky="ew")

        self.netpcsize_label = customtkinter.CTkLabel(self.options_frame_interact, text="Pointcloud input size:", fg_color="transparent", font=text_font)
        self.netpcsize_label.grid(row=7, column=0, padx=10, pady=10, sticky="w")
        self.netpcsize_button = customtkinter.CTkSegmentedButton(self.options_frame_interact, values=["1024", "2048", "4096"], command=self.pcsize_button_callback, font=text_font)
        self.netpcsize_button.set("2048")
        self.netpcsize_button.grid(row=7, column=2, padx=(10, 10), pady=(10, 10), sticky="ew")

        self.bsize_label = customtkinter.CTkLabel(self.options_frame_interact, text="Batch-Size selection:", fg_color="transparent", font=text_font)
        self.bsize_label.grid(row=8, column=0, padx=10, pady=10, sticky="w")
        self.bsize_button = customtkinter.CTkSegmentedButton(self.options_frame_interact, values=["8", "16", "32", "64"], command=self.bsize_button_callback, font=text_font)
        self.bsize_button.set("16")
        self.bsize_button.grid(row=8, column=2, padx=(10, 10), pady=(10, 10), sticky="ew")

        self.train_label = customtkinter.CTkLabel(self.options_frame_interact, text="Train model:", fg_color="transparent", font=text_font)
        self.train_label.grid(row=9, column=0, padx=10, pady=10, sticky="w")
        self.train_checkbox = customtkinter.CTkSwitch(self.options_frame_interact, text="", command=self.train_event, variable=train_var, onvalue="on", offvalue="off")
        self.train_checkbox.grid(row=9, column=2, padx=(10, 10), pady=(10, 10), sticky="w")

        self.tuning_label = customtkinter.CTkLabel(self.options_frame_interact, text="Custom HP-tuning:", fg_color="transparent", font=text_font)
        self.tuning_label.grid(row=10, column=0, padx=10, pady=10, sticky="w")
        self.tuning_checkbox = customtkinter.CTkSwitch(self.options_frame_interact, text="", command=self.tuning_event, variable=tune_var, onvalue="on", offvalue="off")
        self.tuning_checkbox.grid(row=10, column=2, padx=(10, 10), pady=(10, 10), sticky="w")

        self.start_frame_interact = customtkinter.CTkFrame(self.options_frame, fg_color=("gray88", "gray19"))
        self.start_frame_interact.grid(row=8, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
        self.start_frame_interact.grid_columnconfigure(0, weight=1)
        self.start_frame_interact.grid_columnconfigure(1, weight=4)
        self.start_frame_interact.grid_rowconfigure(0, weight=3)
        self.start_frame_interact.grid_rowconfigure(1, weight=1)

        self.mmpnet_start_button = customtkinter.CTkButton(self.start_frame_interact, text="START", font=heading_font, command=self.start_application)
        self.mmpnet_start_button.grid(row=1, column=0, padx=0, pady=10, sticky="nsew")
        self.mmpnet_output = customtkinter.CTkTextbox(self.start_frame_interact, fg_color=("gray90", "gray21"), font=text_font, state="disabled")
        self.mmpnet_output.grid(row=0, column=0, columnspan=2, padx=0, pady=0, sticky="nsew")
        self.progbar_frame = customtkinter.CTkFrame(self.start_frame_interact, fg_color="transparent")
        self.progbar_frame.grid(row=1, column=1, padx=(10, 0), pady=10, sticky="nsew")
        self.progbar_frame.grid_columnconfigure(0, weight=1)
        self.progbar_frame.grid_rowconfigure((0,1,2,3,4), weight=1)
        self.progressbar = customtkinter.CTkProgressBar(self.progbar_frame, orientation="horizontal")
        self.progressbar.grid(row=2, column=0, padx=0, pady=0, sticky="nsew")

        self.appearance_mode_optionemenu.set("System")
        self.scaling_optionemenu.set("100%")
        self.progressbar.configure(determinate_speed=5)
        self.progressbar.set(0)

    def start_application(self):
        self.mmpnet_start_button.configure(state="disabled", text="WORKING...")
        self.progressbar.set(0)
        self.mmpnet_output.configure(state="normal")
        self.mmpnet_output.delete("1.0", "end")
        self.mmpnet_output.configure(state="disabled")
        data_dir = self.folder_select_button_01.cget("text")
        work_dir = self.folder_select_button_02.cget("text")
        model_dir = self.folder_select_button_03.cget("text")
        elimper = float(self.slider_elimper.get())*0.01
        ssstest = float(self.slider_ssstest.get())*0.01
        capsel = self.capsel_button.get()
        growsel = self.growsel_button.get()
        netpcsize = self.netpcsize_button.get()
        netimgsize = 224
        maxpcscale = 0.0075
        bsize = int(self.bsize_button.get())
        train = self.train_checkbox.get()
        tune = self.tuning_checkbox.get()
        thr1 = gui_utils.ReturnValueThread(target=self.run_mmpnet, args=("STARTING MMTSCNET!", data_dir, work_dir, model_dir, elimper, ssstest, capsel, growsel, netpcsize, netimgsize, maxpcscale, bsize, train, self.mmpnet_start_button, self.progressbar, self.mmpnet_output, tune), daemon=True)
        thr1.start()

    def run_mmpnet(self, message, data_dir, work_dir, model_dir, elimper, ssstest, capsel, growsel, netpcsize, netimgsize, maxpcscale, bsize, train, start_btn, progbar, output_log, tune):
        if gui_utils.validate_selected_folders(data_dir, work_dir, model_dir, start_btn, progbar, output_log) == False:
            pass
        else:
            fwf_av = gui_utils.are_fwf_pointclouds_available(data_dir)
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - {message}")
            self.queue_message.put(ticket)
            self.event_generate("<<CheckQueue>>", when="tail")
            self.progressbar.step()

            filepaths = main_functions.extract_data(data_dir, work_dir, output_log, start_btn, progbar, fwf_av)
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - SUCCESSFULLY EXTRACTED ALL FILES!")
            self.queue_message.put(ticket)
            self.event_generate("<<CheckQueue>>", when="tail")
            self.progressbar.step()

            if train == "on":
                X_pc_train, X_pc_val, X_metrics_train, X_metrics_val, X_img_1_train, X_img_1_val, X_img_2_train, X_img_2_val, y_train, y_val, num_classes, label_dict = main_functions.preprocess_data(filepaths, ssstest, capsel, growsel, elimper, maxpcscale, netpcsize, netimgsize, output_log, fwf_av)
                now = datetime.datetime.now()
                now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - TRAINING DATA GENERATED!")
                self.queue_message.put(ticket)
                self.event_generate("<<CheckQueue>>", when="tail")
                self.progressbar.step()

                if tune == "on":
                    untrained_model = main_functions.perform_hp_tuning(model_dir, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, bsize, netpcsize, netimgsize, num_classes, output_log)
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - HYPERPARAMETER TUNING COMPLETED!")
                    self.queue_message.put(ticket)
                    self.event_generate("<<CheckQueue>>", when="tail")
                    self.progressbar.step()

                    trained_model = main_functions.perform_training(untrained_model, bsize, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, model_dir, label_dict, output_log)
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - MMTSCNET TRAINING COMPLETE!")
                    self.queue_message.put(ticket)
                    self.event_generate("<<CheckQueue>>", when="tail")
                    self.progressbar.step()

                    X_pc, X_metrics, X_img_f, X_img_s, y, onehot_to_label_dict, filtered_pointclouds = main_functions.preprocess_prediction_data(filepaths, capsel, growsel, elimper, maxpcscale, netpcsize, netimgsize, output_log, fwf_av)
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - PREDICTION DATA GENERATED!")
                    self.queue_message.put(ticket)
                    self.event_generate("<<CheckQueue>>", when="tail")
                    self.progressbar.step()

                    main_functions.predict_for_custom_data(trained_model, X_pc, X_img_f, X_img_s, X_metrics, y, onehot_to_label_dict, filepaths, work_dir, filtered_pointclouds, output_log)
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - PREDICTING FINISHED!")
                    self.queue_message.put(ticket)
                    self.event_generate("<<CheckQueue>>", when="tail")
                    self.progressbar.step()

                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - MMTSCNET FINISHED!")
                    self.queue_message.put(ticket)
                    self.event_generate("<<CheckQueue>>", when="tail")
                    self.progressbar.step()
                    start_btn.configure(state="normal")

                elif tune == "off":
                    if training_utils.check_if_tuned_model_is_created(model_dir) == True:
                        model_path = training_utils.get_tuned_model_folder(model_dir)
                        untrained_model = training_utils.load_tuned_model_from_folder(model_path)
                        self.progressbar.step()
                        trained_model = main_functions.perform_training(untrained_model, bsize, X_pc_train, X_img_1_train, X_img_2_train, X_metrics_train, y_train, X_pc_val, X_img_1_val, X_img_2_val, X_metrics_val, y_val, model_dir, label_dict, output_log)
                        now = datetime.datetime.now()
                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                        ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - MMTSCNET TRAINING COMPLETE!")
                        self.queue_message.put(ticket)
                        self.event_generate("<<CheckQueue>>", when="tail")
                        self.progressbar.step()

                        X_pc, X_metrics, X_img_f, X_img_s, y, onehot_to_label_dict, filtered_pointclouds = main_functions.preprocess_prediction_data(filepaths, capsel, growsel, elimper, maxpcscale, netpcsize, netimgsize, output_log, fwf_av)
                        now = datetime.datetime.now()
                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                        ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - PREDICTION DATA GENERATED!")
                        self.queue_message.put(ticket)
                        self.event_generate("<<CheckQueue>>", when="tail")
                        self.progressbar.step()

                        main_functions.predict_for_custom_data(trained_model, X_pc, X_img_f, X_img_s, X_metrics, y, onehot_to_label_dict, filepaths, work_dir, filtered_pointclouds, output_log)
                        now = datetime.datetime.now()
                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                        ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - PREDICTING FINISHED!")
                        self.queue_message.put(ticket)
                        self.event_generate("<<CheckQueue>>", when="tail")
                        self.progressbar.step()

                        now = datetime.datetime.now()
                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                        ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - MMTSCNET FINISHED!")
                        self.queue_message.put(ticket)
                        self.event_generate("<<CheckQueue>>", when="tail")
                        self.progressbar.step()
                        start_btn.configure(state="normal")

                    else:
                        now = datetime.datetime.now()
                        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                        ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - ERROR - NO PRETUNED MODEL FOUND!")
                        self.queue_message.put(ticket)
                        self.event_generate("<<CheckQueue>>", when="tail")
                        self.progressbar.step()
                        start_btn.configure(state="normal")

                else:
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - ERROR - AN UNKNOWN ERROR OCCURRED, EXITING!")
                    self.queue_message.put(ticket)
                    self.event_generate("<<CheckQueue>>", when="tail")
                    self.progressbar.step()
                    start_btn.configure(state="normal")

            elif train == "off":
                X_pc, X_metrics, X_img_f, X_img_s, y, onehot_to_label_dict, filtered_pointclouds = main_functions.preprocess_prediction_data(filepaths, capsel, growsel, elimper, maxpcscale, netpcsize, netimgsize, output_log, fwf_av)
                now = datetime.datetime.now()
                now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - PREDICTION DATA GENERATED!")
                self.queue_message.put(ticket)
                self.event_generate("<<CheckQueue>>", when="tail")
                self.progressbar.step()

                if training_utils.check_if_model_is_created(model_dir) == True:
                    model_path = training_utils.get_trained_model_folder(model_dir)
                    trained_model = training_utils.load_trained_model_from_folder(model_path)
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - MMTSCNET LOADED!")
                    self.queue_message.put(ticket)
                    self.event_generate("<<CheckQueue>>", when="tail")
                    self.progressbar.step()

                    main_functions.predict_for_custom_data(trained_model, X_pc, X_img_f, X_img_s, X_metrics, y, onehot_to_label_dict, filepaths, work_dir, filtered_pointclouds, output_log)
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - PREDICTING FINISHED!")
                    self.queue_message.put(ticket)
                    self.event_generate("<<CheckQueue>>", when="tail")
                    self.progressbar.step()

                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - STATUS - MMTSCNET FINISHED!")
                    self.queue_message.put(ticket)
                    self.event_generate("<<CheckQueue>>", when="tail")
                    self.progressbar.step()
                    start_btn.configure(state="normal")

                else:
                    now = datetime.datetime.now()
                    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                    ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - ERROR - NO PRETRAINED MODEL FOUND!")
                    self.queue_message.put(ticket)
                    self.event_generate("<<CheckQueue>>", when="tail")
                    self.progressbar.step()
                    start_btn.configure(state="normal")

            else:
                now = datetime.datetime.now()
                now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
                ticket = gui_utils.Ticket(ticket_type=gui_utils.TicketPurpose.UPDATE_TEXT, ticket_value=f"{now_formatted} - ERROR - AN UNKNOWN ERROR OCCURRED, EXITING!")
                self.queue_message.put(ticket)
                self.event_generate("<<CheckQueue>>", when="tail")
                self.progressbar.step()
                start_btn.configure(state="normal")

    def check_queue(self, event):
        msg: gui_utils.Ticket
        msg = self.queue_message.get()
        if msg.ticket_type == gui_utils.TicketPurpose.UPDATE_TEXT:
            self.mmpnet_output.configure(state="normal")
            self.mmpnet_output.insert("1.0", f"{msg.ticket_value}" + '\n')
            self.mmpnet_output.configure(state="disabled")
        else:
            pass

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def open_website(self):
        url = "https://github.com/jvahrenhold97/MMTSCNet"
        webbrowser.open(url, new=0, autoraise=True)

    def file_select_data(self):
        file_path_data = filedialog.askdirectory()
        if file_path_data:
            self.folder_select_button_01.configure(text=f"{file_path_data}")

    def file_select_work(self):
        file_path_work = filedialog.askdirectory()
        if file_path_work:
            self.folder_select_button_02.configure(text=f"{file_path_work}")

    def file_select_model(self):
        file_path_model = filedialog.askdirectory()
        if file_path_model:
            self.folder_select_button_03.configure(text=f"{file_path_model}")

    def capsel_button_callback(self, value):
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        self.mmpnet_output.configure(state="normal")
        self.mmpnet_output.insert("1.0", f"{now_formatted} - INFO - CAPSEL VALUE: "+ value +'\n')
        self.mmpnet_output.configure(state="disabled")

    def growsel_button_callback(self, value):
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        self.mmpnet_output.configure(state="normal")
        self.mmpnet_output.insert("1.0", f"{now_formatted} - INFO - GROWSEL VALUE: "+ value +'\n')
        self.mmpnet_output.configure(state="disabled")

    def pcsize_button_callback(self, value):
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        self.mmpnet_output.configure(state="normal")
        self.mmpnet_output.insert("1.0", f"{now_formatted} - INFO - PCSISE VALUE: "+ value +'\n')
        self.mmpnet_output.configure(state="disabled")

    def bsize_button_callback(self, value):
        now = datetime.datetime.now()
        now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        self.mmpnet_output.configure(state="normal")
        self.mmpnet_output.insert("1.0", f"{now_formatted} - INFO - BSIZE VALUE: "+ value +'\n')
        self.mmpnet_output.configure(state="disabled")

    def train_event(self):
        if self.train_checkbox.get() == "on":
            self.mmpnet_output.configure(state="normal")
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            self.mmpnet_output.insert("1.0", f"{now_formatted} - INFO - MODEL TRAINING ENABLED" +'\n')
            self.mmpnet_output.configure(state="disabled")
            self.tuning_checkbox.configure(state="normal")
        elif self.train_checkbox.get() == "off":
            self.mmpnet_output.configure(state="normal")
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            self.mmpnet_output.insert("1.0", f"{now_formatted} - INFO - MODEL TRAINING DISABLED" +'\n')
            self.mmpnet_output.configure(state="disabled")
            if self.tuning_checkbox.get() == "on":
                self.tuning_checkbox.toggle()
                self.tuning_checkbox.configure(state="disabled")
        else:
            pass

    def tuning_event(self):
        if self.tuning_checkbox.get() == "on":
            self.mmpnet_output.configure(state="normal")
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            self.mmpnet_output.insert("1.0", f"{now_formatted} - INFO - CUSTOM HP-TUNING ENABLED" +'\n')
            self.mmpnet_output.configure(state="disabled")
        elif self.tuning_checkbox.get() == "off":
            self.mmpnet_output.configure(state="normal")
            now = datetime.datetime.now()
            now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            self.mmpnet_output.insert("1.0", f"{now_formatted} - INFO - CUSTOM HP-TUNING DISABLED" +'\n')
            self.mmpnet_output.configure(state="disabled")
        else:
            pass