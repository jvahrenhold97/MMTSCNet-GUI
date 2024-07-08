import tensorflow as tf
from keras.layers import Input, Conv1D, BatchNormalization, GlobalMaxPooling1D, Dense, Dropout, Concatenate, Reshape, ReLU
from keras.models import Model
from keras.regularizers import L1L2
from keras.applications import DenseNet121
from keras.callbacks import Callback
from sklearn.metrics import f1_score, confusion_matrix
from keras.optimizers import Adam
from keras.utils import Sequence
from keras_tuner import HyperModel, HyperParameters
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def scheduler(epoch, lr):
        if epoch < 7:
            return lr * 1.2
        elif epoch >= 7 and epoch < 20:
            return lr
        else:
            if lr >= 5e-7:
                return lr * 0.95
            else:
                return lr
        
def check_label_corruption(one_hot_labels):
    """
    Check for corruption in one-hot encoded labels.

    Args:
    one_hot_labels (numpy.ndarray): Array of one-hot encoded labels.

    Returns:
    bool: True if corruption is found, False otherwise.
    """
    # Check for NaN values
    if np.isnan(one_hot_labels).any():
        print("One-hot encoded labels contain NaN values.")
        return True

    # Check if each label vector has exactly one element set to 1
    if not np.all(np.sum(one_hot_labels, axis=1) == 1):
        print("One-hot encoded labels are not properly encoded (more than one element set to 1).")
        return True

    # Check if any label vector has all elements set to 0
    if not np.all(np.any(one_hot_labels, axis=1)):
        print("One-hot encoded labels are not properly encoded (no element set to 1).")
        return True

    # No corruption found
    return False

def plot_and_save_history(history, checkpoint_dir, name_param):
    # Create a directory for plots if it doesn't exist
    plots_dir = os.path.join(checkpoint_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/fold" + str(name_param) + "_loss.png"))
    plt.close()
    
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/fold" + str(name_param) + "_accuracy.png"))
    plt.close()

    # Plot training & validation precision values
    plt.figure()
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/fold" + str(name_param) + "_precision.png"))
    plt.close()

    # Plot training & validation recall values
    plt.figure()
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/fold" + str(name_param) + "_recall.png"))
    plt.close()

    # Plot training & validation AUC values
    plt.figure()
    plt.plot(history.history['pr_curve'])
    plt.plot(history.history['val_pr_curve'])
    plt.title('Model area under PR-Curve')
    plt.ylabel('Area under PR-Curve')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/fold" + str(name_param) + "_aucpr.png"))
    plt.close()

    # Plot training & validation AUC values
    plt.figure()
    plt.plot(history.history['pr_at_rec'])
    plt.plot(history.history['val_pr_at_rec'])
    plt.title('Precision at Recall (0.85)')
    plt.ylabel('Precision at Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/fold" + str(name_param) + "_pr_at_rec.png"))
    plt.close()

    # Plot training & validation AUC values
    plt.figure()
    plt.plot(history.history['rec_at_pr'])
    plt.plot(history.history['val_rec_at_pr'])
    plt.title('Recall at Precision (0.85)')
    plt.ylabel('Recall at Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(plots_dir + "/fold" + str(name_param) + "_rec_at_pr.png"))
    plt.close()

def plot_conf_matrix(true_labels, predicted_labels, modeldir, model_file_path):
    y_pred = np.argmax(predicted_labels, axis=1)
    y_true = np.argmax(true_labels, axis=1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(modeldir, model_file_path + '_conf_matrix_predictions_all_data.png'))
    plt.close()

def get_class_distribution(one_hot_labels):
    """
    Returns the class distribution from a set of one-hot encoded labels.
    
    Parameters:
    one_hot_labels (np.ndarray): A 2D NumPy array of one-hot encoded labels.
    
    Returns:
    dict: A dictionary where keys are class indices and values are the counts of each class.
    """
    # Sum along the rows to get the count of each class
    class_counts = np.sum(one_hot_labels, axis=0)
    # Create a dictionary with class indices as keys and counts as values
    class_distribution = {i: int(count) for i, count in enumerate(class_counts)}
    return class_distribution

def map_onehot_to_real(numeric_lbls, numeric_to_text_dict):
    # Ensure numeric_labels is 1D
    numeric_lbls = np.atleast_1d(numeric_lbls)
    text_labels = []
    for numeric in numeric_lbls:
        # Ensure numeric is iterable
        numeric = int(numeric)
        # Get the corresponding text label
        text_label = numeric_to_text_dict.get(numeric)
        if text_label is not None:
            text_labels.append(text_label)
        else:
            raise ValueError(f"Numeric label {numeric} not found in dictionary.")
    return np.array(text_labels)

def check_data(X_train, X_img_1, X_img_2, X_metrics, y_train):
    assert not np.isnan(X_train).any(), "Training pointclouds contain NaN values"
    assert not np.isnan(X_img_1).any(), "Training images (first set) contain NaN values"
    assert not np.isnan(X_img_2).any(), "Training images (second set) contain NaN values"
    assert not np.isnan(X_metrics).any(), "Training metrics contain NaN values"
    assert not np.isnan(y_train).any(), "Training labels contain NaN values"
    assert not np.isinf(X_train).any(), "Training pointclouds contain infinite values"
    assert not np.isinf(X_img_1).any(), "Training images (first set) contain infinite values"
    assert not np.isinf(X_img_2).any(), "Training images (second set) contain infinite values"
    assert not np.isinf(X_metrics).any(), "Training metrics contain infinite values"
    assert not np.isinf(y_train).any(), "Training labels contain infinite values"
    assert np.min(X_train) >= 0 and np.max(X_train) <= 1, "Training pointclouds are not normalized"
    assert np.min(X_img_1) >= 0 and np.max(X_img_1) <= 1, "Training images (first set) are not normalized"
    assert np.min(X_img_2) >= 0 and np.max(X_img_2) <= 1, "Training images (second set) are not normalized"
    assert np.min(X_metrics) >= 0 and np.max(X_metrics) <= 1, "Training metrics are not normalized"
    
def normalize_data(X_pc, X_img_1, X_img_2, X_metrics):
    X_img_1 = X_img_1 / 255.0
    X_img_2 = X_img_2 / 255.0
    scaler_pc = MinMaxScaler()
    X_pc = scaler_pc.fit_transform(X_pc.reshape(-1, X_pc.shape[-1])).reshape(X_pc.shape)
    scaler_metrics = MinMaxScaler()
    X_metrics = scaler_metrics.fit_transform(X_metrics.reshape(-1, X_metrics.shape[-1])).reshape(X_metrics.shape)
    X_pc = np.clip(X_pc, 0, 1)
    X_metrics = np.clip(X_metrics, 0, 1)
    X_img_1 = np.clip(X_img_1, 0, 1)
    X_img_2 = np.clip(X_img_2, 0, 1)
    return X_img_1, X_img_2, X_pc, X_metrics

def generate_class_weights(y_train):
    y_train_int = np.argmax(y_train, axis=1)
    # Get the unique classes
    classes = np.unique(y_train_int)
    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_int)
    # Create a dictionary to pass to the fit method
    return dict(enumerate(class_weights))

class DataGenerator(Sequence):
    def __init__(self, X_pc, X_img_f, X_img_s, X_metrics, y, batch_size):
        self.X_pc = np.array(X_pc)
        self.X_img_f = np.array(X_img_f)
        self.X_img_s = np.array(X_img_s)
        self.X_metrics = np.array(X_metrics)
        self.y = np.array(y)
        self.batch_size = batch_size
        self.indices = np.arange(len(y))
        self.on_epoch_end()

    def __len__(self):
        with tf.device('/CPU:0'):
            return int(np.floor(len(self.y) / float(self.batch_size)))

    def __getitem__(self, index):
        with tf.device('/CPU:0'):
            batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
            X_pc_batch = self.X_pc[batch_indices]
            X_img_f_batch = self.X_img_f[batch_indices]
            X_img_s_batch = self.X_img_s[batch_indices]
            X_metrics_batch = self.X_metrics[batch_indices]
            y_batch = self.y[batch_indices]
            return [X_pc_batch, X_img_f_batch, X_img_s_batch, X_metrics_batch], y_batch

    def on_epoch_end(self):
        with tf.device('/CPU:0'):
            np.random.shuffle(self.indices)

class PointViewGCN(tf.keras.layers.Layer):
    def __init__(self, num_points, hp, **kwargs):
        super(PointViewGCN, self).__init__(**kwargs)
        self.num_points = num_points
        self.hp = hp

    def build(self, input_shape):
        num_conv1d = self.hp.Choice('pvgcn_depth', [1, 2, 3, 4])
        hp_kernel_size = self.hp.Choice('pvgcn_kernel_size', values=[1, 3])
        hp_units = self.hp.Choice('pvgcn_units', values=[256, 512, 1024])
        hp_dropout_rate = self.hp.Float('pvgcn_dropout_rate', min_value=0.2, max_value=0.6, step=0.05)
        hp_regularizer_value = self.hp.Float('pvgcn_regularization', min_value=0.003, max_value=0.01, step=0.001)
        num_neighbors = self.hp.Int('pvgcn_num_neighbors', min_value=8, max_value=64, step=8)
        mlp_first = self.hp.Int('pvgcn_mlp_first', min_value=16, max_value=256, step=16)
        self.transform = TNetLess(input_shape[-1], self.hp, name="t_net")
        self.conv1 = Conv1D(mlp_first, 1, name="pvgcn_conv1d_1")
        self.bnorm1 = BatchNormalization(name="pvgcn_bnorm_1")
        self.relu1 = ReLU(name="pvgcn_relu_1")
        self.dropout1 = Dropout(hp_dropout_rate, name="pvgcn_dropout_1")
        self.conv2 = Conv1D(mlp_first*2, 1, name="pvgcn_conv1d_2")
        self.bnorm2 = BatchNormalization(name="pvgcn_bnorm_2")
        self.relu2 = ReLU(name="pvgcn_relu_2")
        self.dropout2 = Dropout(hp_dropout_rate, name="pvgcn_dropout_2")
        self.maxp1 = GlobalMaxPooling1D(name="pvgcn_maxp_1")
        self.bnorm3 = BatchNormalization(name="pvgcn_bnorm_3")
        self.dense1 = Dense(input_shape[-1], name="pvgcn_dense_1")
        self.concat1 = Concatenate(axis=1, name="pvgcn_concat_1")
        self.conv3 = Conv1D(hp_units, hp_kernel_size, padding='same', kernel_regularizer=L1L2(l1=hp_regularizer_value, l2=hp_regularizer_value), name="pvgcn_conv1d_3")
        self.bnorm4 = BatchNormalization(name="pvgcn_bnorm_4")
        self.relu3 = ReLU(name="pvgcn_relu_3")
        self.dropout3 = Dropout(hp_dropout_rate, name="pvgcn_dropout_3")
        self.conv_blocks = []
        for i in range(1, num_conv1d + 1):
            filters = hp_units // (i * 2)
            conv = Conv1D(filters, hp_kernel_size, padding='same', kernel_regularizer=L1L2(l1=hp_regularizer_value, l2=hp_regularizer_value), name="pvgcn_conv1d_" + str(i+3))
            bnorm = BatchNormalization(name="pvgcn_bnorm_" + str(i+4))
            relu = ReLU(name="pvgcn_relu_" + str(i+3))
            dropout = Dropout(hp_dropout_rate, name="pvgcn_dropout_" + str(i+3))
            self.conv_blocks.append((conv, bnorm, relu, dropout))
        self.maxp2 = GlobalMaxPooling1D(name="pvgcn_maxp_2")
        self.bnorm_globf = BatchNormalization(name="pvgcn_bnorm_globf")

    def call(self, inputs):
        transform = self.transform(inputs)
        point_cloud_transformed = tf.matmul(inputs, transform)
        batch_size = tf.shape(inputs)[0]
        indices = tf.random.uniform((batch_size, self.num_points, self.hp.Int('pvgcn_num_neighbors', min_value=8, max_value=64, step=8)), maxval=self.num_points, dtype=tf.int32)
        features = tf.gather(point_cloud_transformed, indices, axis=1, batch_dims=1)
        features = tf.reshape(features, (batch_size, self.num_points, self.hp.Int('pvgcn_num_neighbors', min_value=8, max_value=64, step=8) * inputs.shape[-1]))
        features = self.conv1(features)
        features = self.bnorm1(features)
        features = self.relu1(features)
        features = self.dropout1(features)
        features = self.conv2(features)
        features = self.bnorm2(features)
        features = self.relu2(features)
        features = self.dropout2(features)
        features = self.maxp1(features)
        features = self.bnorm3(features)
        features = tf.expand_dims(features, axis=1)
        features = tf.tile(features, [1, self.num_points, 1])
        features = self.dense1(features)
        combined_features = self.concat1([point_cloud_transformed, features])
        net = self.conv3(combined_features)
        net = self.bnorm4(net)
        net = self.relu3(net)
        net = self.dropout3(net)
        for conv, bnorm, relu, dropout in self.conv_blocks:
            net = conv(net)
            net = bnorm(net)
            net = relu(net)
            net = dropout(net)
        net = self.maxp2(net)
        global_features = self.bnorm_globf(net)
        return global_features

    def get_config(self):
        config = super(PointViewGCN, self).get_config()
        config.update({
            'num_points': self.num_points,
            'hp': self.hp
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TNetLess(tf.keras.layers.Layer):
    def __init__(self, transform_size, hp, **kwargs):
        super(TNetLess, self).__init__(**kwargs)
        self.transform_size = transform_size
        self.hp = hp

    def build(self, input_shape):
        hp_units_value = self.hp.Choice('t_net_units', values=[8, 16, 32, 64, 128, 256])
        hp_regularizer_value = self.hp.Float('t_net_regularization', min_value=0.003, max_value=0.01, step=0.001)
        hp_dropout_rate_t_net = self.hp.Float('t_net_dropout_rate', min_value=0.2, max_value=0.6, step=0.05)
        self.conv1 = Conv1D(hp_units_value, 1, kernel_regularizer=L1L2(l1=hp_regularizer_value, l2=hp_regularizer_value), name="t_net_conv1d_1")
        self.bnorm1 = BatchNormalization(name="t_net_bnorm_1")
        self.relu1 = ReLU(name="t_net_relu_1")
        self.dropout1 = Dropout(hp_dropout_rate_t_net, name="t_net_dropout_1")
        self.gmaxpool = GlobalMaxPooling1D(name="t_net_gmaxpool")
        self.dense1 = Dense(hp_units_value, kernel_regularizer=L1L2(l1=hp_regularizer_value, l2=hp_regularizer_value), name="t_net_dense_1")
        self.bnorm2 = BatchNormalization(name="t_net_bnorm_2")
        self.relu2 = ReLU(name="t_net_relu_2")
        self.dropout2 = Dropout(hp_dropout_rate_t_net, name="t_net_dropout_2")
        self.dense2 = Dense(self.transform_size**2, activation='linear', kernel_initializer='zeros', bias_initializer='ones', name="t_net_dense_2")
        self.reshape = Reshape((self.transform_size, self.transform_size), name="t_net_reshape")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bnorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.gmaxpool(x)
        x = self.dense1(x)
        x = self.bnorm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.reshape(x)
        return x

    def get_config(self):
        config = super(TNetLess, self).get_config()
        config.update({
            'transform_size': self.transform_size,
            'hp': self.hp
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DenseNetModel(tf.keras.layers.Layer):
    def __init__(self, img_input_shape, model_name, **kwargs):
        super(DenseNetModel, self).__init__(**kwargs)
        self.img_input_shape = img_input_shape
        self._model_name = model_name

    def build(self, input_shape):
        self.model = DenseNet121(include_top=False, input_shape=self.img_input_shape, pooling='avg')

    def call(self, inputs):
        x = self.model(inputs)
        return x

    def get_config(self):
        config = super(DenseNetModel, self).get_config()
        config.update({
            'img_input_shape': self.img_input_shape,
            'model_name': self._model_name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class EnhancedMetricsModel(tf.keras.layers.Layer):
    def __init__(self, hp, **kwargs):
        super(EnhancedMetricsModel, self).__init__(**kwargs)
        self.hp = hp

    def build(self, input_shape):
        units = self.hp.Choice('metrics_units', values=[16, 32, 64, 128, 256])
        dropout_rate = self.hp.Float('metrics_dropout_rate', min_value=0.2, max_value=0.6, step=0.05)
        regularization = self.hp.Float('metrics_regularization', min_value=0.003, max_value=0.01, step=0.001)

        self.dense1 = Dense(units, activation='relu', kernel_regularizer=L1L2(l1=regularization, l2=regularization), name="metrics_dense_1")
        self.bnorm1 = BatchNormalization(name="metrics_bnorm_1")
        self.dropout1 = Dropout(dropout_rate, name="metrics_dropout_1")
        self.dense2 = Dense(units // 2, activation='relu', kernel_regularizer=L1L2(l1=regularization, l2=regularization), name="metrics_dense_2")
        self.bnorm2 = BatchNormalization(name="metrics_bnorm_2")
        self.dropout2 = Dropout(dropout_rate, name="metrics_dropout_2")
        self.dense3 = Dense(units // 4, activation='relu', kernel_regularizer=L1L2(l1=regularization, l2=regularization), name="metrics_dense_3")
        self.bnorm3 = BatchNormalization(name="metrics_bnorm_3")
        self.dropout3 = Dropout(dropout_rate, name="metrics_dropout_3")
        self.dense4 = Dense(units // 8, activation='relu', kernel_regularizer=L1L2(l1=regularization, l2=regularization), name="metrics_dense_4")
        self.bnorm4 = BatchNormalization(name="metrics_bnorm_4")
        self.dropout4 = Dropout(dropout_rate, name="metrics_dropout_4")

        super(EnhancedMetricsModel, self).build(input_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.bnorm1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.bnorm2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.bnorm3(x)
        x = self.dropout3(x)
        x = self.dense4(x)
        x = self.bnorm4(x)
        x = self.dropout4(x)
        return x

    def get_config(self):
        config = super(EnhancedMetricsModel, self).get_config()
        config.update({
            'hp': self.hp
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CombinedModel(HyperModel):
    def __init__(self, point_cloud_shape, image_shape, metrics_shape, num_classes, num_points, **kwargs):
        super(CombinedModel, self).__init__(**kwargs)
        self.point_cloud_shape = point_cloud_shape
        self.image_shape = image_shape
        self.metrics_shape = metrics_shape
        self.num_classes = num_classes
        self.num_points = num_points

    def build(self, hp):
        pointnet_input = Input(shape=self.point_cloud_shape, name='pointnet_input')
        image_input_1 = Input(shape=self.image_shape, name='image_input_1')
        image_input_2 = Input(shape=self.image_shape, name='image_input_2')
        metrics_input = Input(shape=self.metrics_shape, name='metrics_input')

        pointnet_branch = PointViewGCN(self.num_points, hp)(pointnet_input)
        image_branch_1 = DenseNetModel(self.image_shape, "image_branch_1")(image_input_1)
        image_branch_2 = DenseNetModel(self.image_shape, "image_branch_2")(image_input_2)
        metrics_branch = EnhancedMetricsModel(hp, input_shape=self.metrics_shape)(metrics_input)

        concatenated = Concatenate(name="concat_all")([pointnet_branch, image_branch_1, image_branch_2, metrics_branch])
        x = concatenated

        num_dense = hp.Choice('clss_depth', [1, 2, 3, 4, 5])
        units_dense = hp.Choice('clss_units', [120, 240, 330, 480, 600])
        dropout_clss = hp.Float('clss_dropout_rate', min_value=0.2, max_value=0.6, step=0.05)
        regularizer_value_clss = hp.Float('clss_regularization', min_value=0.003, max_value=0.01, step=0.001)

        for i in range(1, num_dense + 1):
            units = int(units_dense/i)
            x = Dense(units, kernel_regularizer=L1L2(l1=regularizer_value_clss, l2=regularizer_value_clss), name="clss_dense_" + str(i))(x)
            x = BatchNormalization(name="clss_bnorm_" + str(i))(x)
            if i < num_dense + 1:
                x = ReLU(name="clss_relu_" + str(i))(x)
                x = Dropout(dropout_clss, name="clss_dropout_" + str(i))(x)
            else:
                x = ReLU(name="clss_relu_" + str(i))(x)

        output = Dense(self.num_classes, activation='softmax', name='output')(x)

        model = Model(inputs=[pointnet_input, image_input_1, image_input_2, metrics_input], outputs=output)

        initial_learning_rate = hp.Choice('learning_rate', [1e-5, 5e-6, 1e-6, 5e-7])
        model.compile(optimizer=Adam(learning_rate=initial_learning_rate, clipnorm=1.0),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="pr_curve", curve="PR"), tf.keras.metrics.PrecisionAtRecall(0.85, name="pr_at_rec"), tf.keras.metrics.RecallAtPrecision(0.85, name="rec_at_pr")])
        return model

    def get_untrained_model(self, best_hyperparameters):
        return self.build(best_hyperparameters)

    def get_config(self):
        config = super(CombinedModel, self).get_config()
        config.update({
            'point_cloud_shape': self.point_cloud_shape,
            'image_shape': self.image_shape,
            'metrics_shape': self.metrics_shape,
            'num_classes': self.num_classes,
            'num_points': self.num_points
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class MacroF1ScoreCallback(Callback):
    def __init__(self, validation_data, batch_size):
        super().__init__()
        self.validation_data = validation_data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        val_gen = self.validation_data
        y_true = []
        y_pred = []
        
        for i in range(len(val_gen)):
            X_val, y_val = val_gen[i]
            y_true.extend(np.argmax(y_val, axis=1))
            y_pred.extend(np.argmax(self.model.predict(X_val, batch_size=self.batch_size, verbose=0), axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        logs['val_macro_f1'] = macro_f1
        print(f" — val_macro_f1: {macro_f1:.4f}")