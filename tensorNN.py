import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Dropout
from keras import Sequential
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def compute_class_weights(y):
    class_labels = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y)
    class_weights_dict = {class_labels[i]: class_weights[i] for i in range(len(class_labels))}
    return class_weights_dict

def plot_boxplot_with_outliers(X):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=X, palette='viridis', showfliers=True)
    plt.title('Boxplot of Features with Outliers')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.show()

def preprocess_data(df, corr_threshold):
    df = df.drop_duplicates().dropna()

    # Plotting the data with outliers before applying the correlation threshold
    plot_boxplot_with_outliers(df)  

    target_column = df.columns[-1]
    corr_matrix = df.corr()
    high_corr_columns = corr_matrix[target_column][abs(corr_matrix[target_column]) > corr_threshold].index
    print(f"Columns used in the model based on the correlation threshold: {high_corr_columns.tolist()}")

    # Keeping only high correlation columns
    df = df[high_corr_columns]

    # Calculating the IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Removing outliers
    df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df_no_outliers


def plot_class_distribution(y):
    # Encode labels for better visualization
    encoded_labels = y.copy()
    encoded_labels[encoded_labels <= 4] = 0
    encoded_labels[encoded_labels == 5] = 1
    encoded_labels[encoded_labels == 6] = 2
    encoded_labels[encoded_labels >= 7] = 3

    plt.figure(figsize=(8, 5))
    sns.countplot(x=encoded_labels, palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Encoded Quality')
    plt.ylabel('Count')
    plt.show()



def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

def encode_labels(y):
    encoded_y = y.copy()
    encoded_y[y <= 4] = 0
    encoded_y[y == 5] = 1
    encoded_y[y == 6] = 2
    encoded_y[y >= 7] = 3
    return encoded_y

def create_model(input_shape, learning_rate=0.001, num_hidden_layers=3, num_neurons=128, dropout_rate=0.5, l2_regularization=0.01):
    model = Sequential()

    model.add(Dense(num_neurons, activation='relu', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)))
    model.add(Dropout(dropout_rate))

    for _ in range(num_hidden_layers - 1):
        model.add(Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(4, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(X_train, y_train, X_val, y_val, class_weights, input_shape, learning_rate=0.001, batch_size=10):
    model = create_model(input_shape=input_shape, learning_rate=learning_rate)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        class_weight=class_weights,
        epochs=500,
        batch_size=batch_size,
        verbose=0,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stop]
    )
    
    return model, history

def plot_training_history(history):
        # Plot the training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    # Loading the dataset
    df = pd.read_csv("winequality-red.csv", delimiter=';')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Plot the class distribution
    plot_class_distribution(y)

    # Preprocess the data and plot outliers
    df_processed = preprocess_data(df, corr_threshold=0.15)
    
    # After preprocessing, plot the correlation matrix
    plot_correlation_matrix(df_processed)

    # Encoding the labels
    y_encoded = encode_labels(y)

    # Scaling the features
    X_scaled = StandardScaler().fit_transform(X)

    # Compute class weights
    class_weights = compute_class_weights(y_encoded)

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Determine the input shape for the neural network
    input_shape = (X_train.shape[1],)

    # Train the model
    best_model, history = train_model(X_train, y_train, X_val, y_val, class_weights, input_shape, learning_rate=0.001, batch_size=10)

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test).argmax(axis=1)

    # Plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    plot_training_history(history)

    new_data_list = [
        [8.1, 0.38, 0.28, 2.1, 0.066, 13, 30, 0.9968, 3.23, 0.73, 9.7],
        [7.6, 0.51, 0.15, 2.8, 0.11, 33, 73, 0.9955, 3.17, 0.63, 10.2],
        [10.2, 0.42, 0.57, 3.4, 0.07, 4, 10, 0.9971, 3.04, 0.63, 9.5],
        [7.4, 0.59, 0.08, 4.4, 0.086, 6, 29, 0.9974, 3.38, 0.5, 9]
    ]

    new_data_array = np.array(new_data_list)

    predictions = best_model.predict(new_data_array)
    predicted_classes = np.argmax(predictions, axis=1)

    quality_levels = {
        0: "χαμηλής ποιότητας",  # Low Quality
        1: "κατώτερης μέτριας ποιότητας",  # Lower Middle Quality
        2: "ανώτερης μέτριας ποιότητας",  # Upper Middle Quality
        3: "ανώτερης ποιότητας"  # High Quality
    }

    for i, predicted_class in enumerate(predicted_classes):
        quality_description = quality_levels.get(predicted_class, "Unknown Class")
        print(f"Prediction for data {i+1}: {quality_description}")

    # Print the accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Set: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
