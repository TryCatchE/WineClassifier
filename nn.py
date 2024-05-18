import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_class_distribution(y):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y, palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Encoded Quality')
    plt.ylabel('Count')
    plt.show()

def visualize_correlation(X, y):
    # Concatenate X and y to include the target variable in the correlation matrix
    data_with_target = pd.concat([X, y], axis=1)

    # Display heatmap of the correlation matrix
    correlation_matrix = data_with_target.corr()

    # Print correlation values
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.show()

def remove_outliers_for_classes(df, classes_to_filter):
    for cls in classes_to_filter:
        class_data = df[df['quality'] == cls]
        Q1 = class_data.quantile(0.25)
        Q3 = class_data.quantile(0.75)
        IQR = Q3 - Q1
        df = df.drop(class_data[~((class_data >= (Q1 - 1.5 * IQR)) & (class_data <= (Q3 + 1.5 * IQR))).all(axis=1)].index)
    return df

def encode_labels(y):
    def encode(label):
        if label <= 4:
            return 0
        elif label == 5:
            return 1
        elif label == 6:
            return 2
        elif label >= 7:
            return 3

    return y.apply(encode)


def load_and_split_data(file_path, correlation_threshold):
    data = pd.read_csv(file_path, delimiter=';')

    # Remove duplicates and rows with NaN values
    # data = data.drop_duplicates().dropna()

    # # Identify classes with more data
    # class_counts = data['quality'].value_counts()
    # classes_with_more_data = class_counts[class_counts > class_counts.median()].index.tolist()

    # Remove outliers from these classes
    # data = remove_outliers_for_classes(data, classes_with_more_data)

    features = data.iloc[:, :-1]

    labels = encode_labels(data.iloc[:, -1])
    print(labels)
    print(data.iloc[:, -1])
    exit()

    # Compute correlations and filter features
    correlation_with_labels = features.corrwith(labels).abs()
    filtered_features = correlation_with_labels[correlation_with_labels >= correlation_threshold].index
    filtered_features_data = features[filtered_features]

    return train_test_split(filtered_features_data, labels, test_size=0.2, random_state=42, stratify=labels)

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the MLPClassifier
    neural_network = MLPClassifier(random_state=42)

    # Hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': [(20, 20) ,(28, 28,28)],
        'activation': ['relu', 'tanh', 'logistic'],
        'max_iter': [100000],
        'learning_rate_init': [0.001, 0.01],
        'learning_rate': ['adaptive'],
        'solver': [ 'sgd'],
        'early_stopping': [True],
        'momentum': [0.9, 0.99], 
    }
   
    # GridSearchCV
    f1 = make_scorer(f1_score , average='macro')
    grid_search = GridSearchCV(neural_network, param_grid, cv=5, scoring=f1, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test_scaled)

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Plot the confusion matrix
    plot_confusion_matrix(cm, classes=['0', '1', '2', '3'], normalize=True, title='Normalized Confusion Matrix')

    return grid_search.best_params_, accuracy, report, best_model, X_train_scaled


def main():
    
    X_train, X_test, y_train, y_test = load_and_split_data("winequality-red.csv", correlation_threshold=0.1)
    
    plot_class_distribution(y_test)
    visualize_correlation(X_train, y_train)
    
    best_params, accuracy, report,best_model, X_train_scaled = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    print(f"Best Parameters: {best_params}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
        
    # plot_learning_curve(best_model, "Learning Curve", X_train_scaled, y_train, cv=3)

    plt.show()
    # Make predictions with all the test data
    # predictions = best_model.predict(X_test)

    # # Create a DataFrame for easier comparison
    # results_df = pd.DataFrame({
    #     'Actual': y_test,
    #     'Predicted': predictions
    # })
    # results_df['Correct'] = results_df['Actual'] == results_df['Predicted']

    # print(results_df)

    # # Count correct predictions
    # correct_predictions_count = results_df['Correct'].sum()
    # print(f"\nTotal Correct Predictions: {correct_predictions_count} out of {len(y_test)}")

if __name__ == "__main__":
    main()
