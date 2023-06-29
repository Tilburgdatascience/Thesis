#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Sequence length plot creation
import matplotlib.pyplot as plt
# Summary statistics
print("Mean sequence length:", np.mean(sequence_lengths))
print("Median sequence length:", np.median(sequence_lengths))
print("Maximum sequence length:", np.max(sequence_lengths))
print("Standard deviation of sequence lengths:", np.std(sequence_lengths))

# Histogram
plt.hist(sequence_lengths, bins=20)
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.title("Distribution of Sequence Lengths")

# Set x-axis range
max_seq_len = np.max(sequence_lengths)
plt.xlim(0, max_seq_len)

plt.show()


# In[ ]:


#DistilBERT model
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
import pandas as pd
import nltk
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only required once)
nltk.download('stopwords')

# Preprocessing functions
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Load and preprocess data
data = pd.read_csv("/home/u953292/Projects/data/Thesisdata.csv")
encoding = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
labels = ['Negative', 'Neutral', 'Positive']
X = data['Review'].copy()
y = data['Rating'].map(encoding)

# Tokenization and encoding
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
seq_len = 512
encodings = tokenizer(X.tolist(), truncation=True, padding=True, max_length=seq_len)

input_ids = np.array(encodings['input_ids'])
attention_mask = np.array(encodings['attention_mask'])

# BERT model
bert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
inp_ids = L.Input(shape=(seq_len,), dtype=tf.int32)
att_mask = L.Input(shape=(seq_len,), dtype=tf.int32)
last_hidden_state = bert(inp_ids, attention_mask=att_mask)[0]
out = last_hidden_state[:, 0, :]
bert_model = tf.keras.Model(inputs=[inp_ids, att_mask], outputs=out)

# Model architecture
model = tf.keras.Sequential([
    L.Input(shape=(768,)),
    L.Dense(256, activation='relu'),
    L.Dropout(0.5),
    L.Dense(3, activation='softmax')
])

# compile the model
model.compile(loss=SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

# K-fold cross-validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)
accuracy_scores = []
f1_scores = []
mae_scores = []
rmse_scores = []
confusion_matrices = []

for train_index, val_index in kf.split(input_ids):
    # Split data into train and validation sets
    input_ids_train, input_ids_val = input_ids[train_index], input_ids[val_index]
    attention_mask_train, attention_mask_val = attention_mask[train_index], attention_mask[val_index]
    train_labels, val_labels = y[train_index], y[val_index]

    # Extract BERT features
    bert_output_train = bert_model.predict([input_ids_train, attention_mask_train], batch_size=16, verbose=1)
    bert_output_val = bert_model.predict([input_ids_val, attention_mask_val], batch_size=16, verbose=1)

    # Train the model
    history = model.fit(bert_output_train, train_labels, epochs=22, batch_size=32, verbose=1)

    # Evaluate the model on the validation set
    val_pred = np.argmax(model.predict(bert_output_val), axis=1)
    val_accuracy = accuracy_score(val_pred, val_labels)
    val_f1 = f1_score(val_pred, val_labels, average='macro')
    val_mae = mean_absolute_error(val_pred, val_labels)
    val_rmse = mean_squared_error(val_pred, val_labels, squared=False)
    val_confusion_matrix = confusion_matrix(val_labels, val_pred)

    accuracy_scores.append(val_accuracy)
    f1_scores.append(val_f1)
    mae_scores.append(val_mae)
    rmse_scores.append(val_rmse)
    confusion_matrices.append(val_confusion_matrix)

# Calculate and print the average metrics
average_accuracy = np.mean(accuracy_scores)
average_f1 = np.mean(f1_scores)
average_mae = np.mean(mae_scores)
average_rmse = np.mean(rmse_scores)
average_confusion_matrix = np.mean(confusion_matrices, axis=0)

print('Average Accuracy: {}'.format(average_accuracy))
print('Average F1 Score: {}'.format(average_f1))
print('Average MAE: {}'.format(average_mae))
print('Average RMSE: {}'.format(average_rmse))
print('Average Confusion Matrix:\n{}'.format(average_confusion_matrix))


# In[12]:


#Random Forest model
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load and preprocess data
data = pd.read_csv("C:/Users/31611/Downloads/Thesisdata.csv")
encoding = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
labels = ['Negative', 'Neutral', 'Positive']
X = data['Review'].copy()
y = data['Rating'].map(encoding)

# Preprocessing functions
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Apply preprocessing to the data
X_preprocessed = X.apply(preprocess_text)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the preprocessed text
X_features = vectorizer.fit_transform(X_preprocessed)

# Train Random Forest model
model = RandomForestClassifier()

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=67)
accuracy_scores = []
f1_scores = []
mae_scores = []
rmse_scores = []
confusion_matrices = []

for train_index, test_index in kf.split(X_features):
    # Split data into train and test sets for cross-validation
    train_features, test_features = X_features[train_index], X_features[test_index]
    train_labels, test_labels = y[train_index], y[test_index]
    
    # Fit the model
    model.fit(train_features, train_labels)
    
    # Predict on the test set
    pred = model.predict(test_features)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels, pred)
    f1 = f1_score(test_labels, pred, average='weighted')
    mae = mean_absolute_error(test_labels, pred)
    rmse = np.sqrt(mean_squared_error(test_labels, pred))
    confusion = confusion_matrix(test_labels, pred)
    
    # Append scores and confusion matrix to lists
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    confusion_matrices.append(confusion)

# Calculate average scores
avg_accuracy = np.mean(accuracy_scores)
avg_f1 = np.mean(f1_scores)
avg_mae = np.mean(mae_scores)
avg_rmse = np.mean(rmse_scores)

# Print evaluation results
print("Average Accuracy:", avg_accuracy)
print("Average F1 Score:", avg_f1)
print("Average MAE:", avg_mae)
print("Average RMSE:", avg_rmse)

cm = confusion_matrix(test_labels, pred, labels=[0, 1, 2])

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')


plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[22]:


#MultinomialNB model
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load and preprocess data
data = pd.read_csv("C:/Users/31611/Downloads/Thesisdata.csv")
encoding = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
labels = ['Negative', 'Neutral', 'Positive']
X = data['Review'].copy()
y = data['Rating'].map(encoding)

# Preprocessing functions
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Apply preprocessing to the data
X_preprocessed = X.apply(preprocess_text)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the preprocessed text
X_features = vectorizer.fit_transform(X_preprocessed)

# Train model
model = MultinomialNB()

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=67)
accuracy_scores = []
f1_scores = []
mae_scores = []
rmse_scores = []
confusion_matrices = []

for train_index, test_index in kf.split(X_features):
    # Split data into train and test sets for cross-validation
    train_features, test_features = X_features[train_index], X_features[test_index]
    train_labels, test_labels = y[train_index], y[test_index]
    
    # Fit the model
    model.fit(train_features, train_labels)
    
    # Predict on the test set
    pred = model.predict(test_features)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels, pred)
    f1 = f1_score(test_labels, pred, average='weighted')
    mae = mean_absolute_error(test_labels, pred)
    rmse = np.sqrt(mean_squared_error(test_labels, pred))
    confusion = confusion_matrix(test_labels, pred)
    
    # Append scores and confusion matrix to lists
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    confusion_matrices.append(confusion)

# Calculate average scores
avg_accuracy = np.mean(accuracy_scores)
avg_f1 = np.mean(f1_scores)
avg_mae = np.mean(mae_scores)
avg_rmse = np.mean(rmse_scores)

# Print evaluation results
print("Average Accuracy:", avg_accuracy)
print("Average F1 Score:", avg_f1)
print("Average MAE:", avg_mae)
print("Average RMSE:", avg_rmse)

cm = confusion_matrix(test_labels, pred, labels=[0, 1, 2])

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')


plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[42]:


#Perceptron model
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load and preprocess data
data = pd.read_csv("C:/Users/31611/Downloads/Thesisdata.csv")
encoding = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
labels = ['Negative', 'Neutral', 'Positive']
X = data['Review'].copy()
y = data['Rating'].map(encoding)

# Preprocessing functions
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Apply preprocessing to the data
X_preprocessed = X.apply(preprocess_text)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the preprocessed text
X_features = vectorizer.fit_transform(X_preprocessed)

# Train model
model = Perceptron()

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=67)
accuracy_scores = []
f1_scores = []
mae_scores = []
rmse_scores = []
confusion_matrices = []

for train_index, test_index in kf.split(X_features):
    # Split data into train and test sets for cross-validation
    train_features, test_features = X_features[train_index], X_features[test_index]
    train_labels, test_labels = y[train_index], y[test_index]
    
    # Fit the model
    model.fit(train_features, train_labels)
    
    # Predict on the test set
    pred = model.predict(test_features)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels, pred)
    f1 = f1_score(test_labels, pred, average='weighted')
    mae = mean_absolute_error(test_labels, pred)
    rmse = np.sqrt(mean_squared_error(test_labels, pred))
    confusion = confusion_matrix(test_labels, pred)
    
    # Append scores and confusion matrix to lists
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    confusion_matrices.append(confusion)

# Calculate average scores
avg_accuracy = np.mean(accuracy_scores)
avg_f1 = np.mean(f1_scores)
print(f1_scores)
avg_mae = np.mean(mae_scores)
avg_rmse = np.mean(rmse_scores)

# Print evaluation results
print("Average Accuracy:", avg_accuracy)
print("Average F1 Score:", avg_f1)
print("Average MAE:", avg_mae)
print("Average RMSE:", avg_rmse)

cm = confusion_matrix(test_labels, pred, labels=[0, 1, 2])

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')


plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[53]:


# DistilBERT confusion matrix
import numpy as np
import matplotlib.pyplot as plt

# Existing confusion matrix
cm = np.array([[494, 75, 91],
               [108, 115, 216],
               [63, 116, 2820]])

# Calculate precision, recall, and F1 score
tp = np.diag(cm)
fp = np.sum(cm, axis=0) - tp
fn = np.sum(cm, axis=1) - tp

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
print(f1)
average_f1 = np.mean(f1)

# Print precision, recall, and average F1 score
print("Precision:", precision)
print("Recall:", recall)
print("Average F1-score:", average_f1)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7.5, 7.5))
im = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

