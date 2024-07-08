from gui.main_window import App
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(gpus, logical_gpus)

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()