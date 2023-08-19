#!/usr/bin/env python
# coding: utf-8

# ## CAR LOAN APPROVAL USING MACHINE LEARNING AND DATA VISULAIZATION

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import sklearn.linear_model, sklearn.datasets # sklearn is an important package for much of the ML we will be doing, this time we are using the Linear Regression Model and the datasets
from sklearn import kernel_ridge 
print ("Ready")


# In[2]:


from IPython.display import HTML
def pretty_print_data(value_counts_):
  "Quick function to display value counts more nicely"
  display(HTML(pd.DataFrame(value_counts_).to_html()))


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
sns.set()
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore')


# In[4]:


data = pd.read_csv("LoanApproval.csv")
data_copy = data
print(data_copy.size)
data.head(5)


# #### Data Preprocessing

# In[5]:


obj = (data.dtypes == 'object')
print("Categorical variables:",len(list(obj[obj].index)))


# In[6]:


# shape of dataset: 
print(data.shape)

# list of column titles 
print(data.columns)


# In[7]:


# Dropping Loan_ID column
data.drop(['Loan_ID'],axis=1,inplace=True)
data.isna().sum()


# In[8]:


# list of column (field) data types
print(data.dtypes)


# # Exploratory data analysis

# In[9]:


# Summary statistics for numerical features
data.describe()


# In[10]:


# let's list all categorical features #Univariate Analysis: involves exploring individual features in isolation to understand their distribution and characteristics
categorical_columns= [ 'Gender', 
                      'Married', 'Education', 'Self_Employed', 'Property_Area', 
                      'Loan_Status']

# let's get the categories and their count for each feature
for col in categorical_columns:
  print(f"Categories and number of occurrences for '{col}'")
  pretty_print_data(data[col].value_counts())
  print()


# ### Managing missing data

# In[11]:


data.isna().sum()


# ## Handling missing data

# ### Simple Imputer (numerical value)

# In[12]:


# handling missing data
from sklearn.impute import SimpleImputer 

data_no_nan =  data.copy()

# 1. Imputer
imptr_num = SimpleImputer(missing_values = np.nan, strategy = 'median')  

# 2. Fit the imputer object to the feature matrix (only for numeric features)
numerical_columns = ['Dependents', 'LoanAmount','Loan_Amount_Term', 'Credit_History']
imptr_num = imptr_num.fit(data_no_nan[numerical_columns]) # fit the data to estimate the parameters (here, the average value)

# 3. Call Transform to replace missing data in train_dataset (on specific columns) by the mean of the column to which that missing data belongs to
data_no_nan[numerical_columns] = \
  imptr_num.transform(data_no_nan[numerical_columns]) # apply the transformation using the parameters estimated above

# note column ApplicantIncome in the first row --> before it was a missing value!
data_no_nan.head()


# In[13]:


data_no_nan.isna().sum()


# ## Removing duplicate rows

# In[14]:


# Check for duplicated rows
duplicates = data_no_nan.duplicated()

# Count the number of duplicated rows
num_duplicates = duplicates.sum()

# Print the number of duplicated rows
print("Number of duplicated rows: ", num_duplicates)


# In[15]:


print(f'Original dataset length: {len(data_no_nan)}')
duplicate_data = data_no_nan.drop_duplicates()
cleaned_data=duplicate_data
print(f'Dataset length after removing duplicate rows: {len(duplicate_data)}')
print()
cleaned_data.head()


# # Visualize class distribution

# In[16]:


# Check for class distribution
print("Class distribution:\n", cleaned_data['Loan_Status'].value_counts())


# In[17]:


# Visualize class distribution
sns.countplot(x='Loan_Status', data=cleaned_data)
plt.show()


# In[18]:


obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
plt.figure(figsize=(18,36))
index = 1
  
for col in object_cols:
  y = data[col].value_counts()
  plt.subplot(11,4,index)
  plt.xticks(rotation=90)
  sns.barplot(x=list(y.index), y=y)
  index +=1


# Visualize all the unique values in columns using barplot. This will simply show which value is dominating as per our dataset.

# In[19]:


# Bivariate Analysis: Scatter Plots
plt.figure(figsize=(10, 6))
plt.scatter(cleaned_data['ApplicantIncome'], cleaned_data['LoanAmount'], cmap='coolwarm', s=75, edgecolors='k')
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.title('Scatter Plot: Applicant Income vs. Loan Amount')
plt.colorbar(label='Loan Status (0: Rejected, 1: Approved)')
plt.show()


# Visualizing the relationship between numerical features and the target variable. 
# In this case, we'll plot 'ApplicantIncome' and 'LoanAmount' against 'Loan_Status':

# In[20]:


# Bivariate Analysis: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cleaned_data.corr(), cmap='coolwarm', annot=True, fmt='.2f', linewidths=1)
plt.title('Correlation Heatmap')
plt.show()


# A correlation heatmap provides a visual representation of the correlations between numerical features. This will help you identify which features are more strongly related to each other and their potential influence on loan approval:

# # DATA PRE-PROCESSING AND VISUALIZATION

# # Encoding the Categorical Data

# In[21]:


from sklearn.preprocessing import LabelEncoder
lblEncoder_X= LabelEncoder()


# In[22]:


# create a new column 'light_conditions_encoded' and use .loc to assign the values
cleaned_data.loc[:, 'Gender'] = lblEncoder_X.fit_transform(cleaned_data['Gender'])
cleaned_data.loc[:, 'Married'] = lblEncoder_X.fit_transform(cleaned_data['Married'])
cleaned_data.loc[:, 'Education'] = lblEncoder_X.fit_transform(cleaned_data['Education'])
cleaned_data.loc[:, 'Self_Employed'] = lblEncoder_X.fit_transform(cleaned_data['Self_Employed'])
cleaned_data.loc[:, 'Property_Area'] = lblEncoder_X.fit_transform(cleaned_data['Property_Area'])
cleaned_data.loc[:, 'Loan_Status'] = lblEncoder_X.fit_transform(cleaned_data['Loan_Status'])


# In[23]:


cleaned_data.head()       


# #### Separate features and target variable

# In[24]:


# Separate features and target variable
X = cleaned_data.drop(['Loan_Status'], axis=1)
y = cleaned_data['Loan_Status']


# ##### Feature Scaling

# In[25]:


# 3. Scale the data using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# #### OverSampling

# In[26]:


get_ipython().system('pip install imblearn')
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)


# ##### Train Test Split

# In[27]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


# To find the number of columns with 
# datatype==object
obj = (data.dtypes == 'object')
print("Categorical variables:",len(list(obj[obj].index)))


# In[29]:


plt.figure(figsize=(12,6))
  
sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f',
            linewidths=2,annot=True)


# # DIMENSIONALITY REDUCTION

# In[30]:


from sklearn.decomposition import PCA


# In[31]:


pca = PCA(n_components=2)  # Choose the number of components you want to keep
X_pca = pca.fit_transform(X)

# Plot PCA
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k', s=75)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA plot')
plt.colorbar()
plt.show()


# # CLASSIFICATION

# # CREATING A FUNCTION TO EVALUATE THE MODEL PERFORMANCE

# In[32]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print performance metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Print classification report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Create confusion matrix and display it as a heatmap
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# # LOGISTIC REGRESSION

# In[33]:


import warnings
from sklearn.model_selection import GridSearchCV

warnings.simplefilter("ignore")

param_grid = {
    'C': np.logspace(-4, 4, 20),  
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
    'max_iter': [20000],
    'class_weight': ['balanced']
}

lr = LogisticRegression()

grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=10, scoring='accuracy', refit=True)
grid = grid.fit(X_train, y_train)

print('Best estimator: {}\nWeights: {}, Intercept: {}\nBest params: {}'.format(
    grid.best_estimator_, grid.best_estimator_.coef_, grid.best_estimator_.intercept_, grid.best_params_))


# In[34]:


log_reg = grid.best_estimator_
print("Logistic Regression Results:")
# Evaluate the model using the evaluate_model_performance function
log_reg_results = evaluate_model_performance(log_reg, X_test, y_test)


# # DECISION TREE CLASSIFIER

# In[35]:


import warnings
warnings.simplefilter("ignore")
param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_features": ["auto","sqrt", "log2"], "splitter":["best", "random"]
}

rf = DecisionTreeClassifier()

grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='accuracy', refit=True)
grid = grid.fit(X_train, y_train)

print('Best estimator:', grid.best_estimator_)
print('Best params:', grid.best_params_)
print('Best Score:', grid.best_score_)

# get feature importance of the best estimator
feature_importance = grid.best_estimator_.feature_importances_
print('Feature importance:', feature_importance)


# In[36]:


dt = grid.best_estimator_
print("Decision Tree Results:")
dt_results = evaluate_model_performance(dt, X_test, y_test)


# # RANDOM FOREST

# In[37]:


param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "n_estimators": [10, 50, 100, 200],
    "max_features": ["sqrt", "log2"]
}

rf = RandomForestClassifier(n_estimators=100)

grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='accuracy', refit=True)
grid = grid.fit(X_train, y_train)

print('Best estimator:', grid.best_estimator_)
print('Best params:', grid.best_params_)

# get feature importance of the best estimator
feature_importance = grid.best_estimator_.feature_importances_
print('Feature importance:', feature_importance)


# In[38]:


# Random Forest
rf = grid.best_estimator_
rf.fit(X_train, y_train)
print("Random Forest Results:")
rf_results = evaluate_model_performance(rf, X_test, y_test) 


# # K-Nearest Neighbors

# In[39]:


param_grid = {
    "n_neighbors": list(range(1, 31)),  
    "weights": ["uniform", "distance"],  
    "metric": ["euclidean", "manhattan", "minkowski"] 
}

knn = KNeighborsClassifier()

grid = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10, scoring='accuracy', refit=True)
grid = grid.fit(X_train, y_train)

print('Best estimator:', grid.best_estimator_)
print('Best params:', grid.best_params_)
print('Best Score:', grid.best_score_)


# In[40]:


# K-Nearest Neighbors
knn = grid.best_estimator_
knn.fit(X_train, y_train)
print("K-Nearest Neighbors Results:")
knn_results = evaluate_model_performance(knn, X_test, y_test)


# # Naive Bayes

# In[41]:


from sklearn.naive_bayes import GaussianNB
param_grid = {
    "var_smoothing": np.logspace(-10, -1, 10) 
}

gnb = GaussianNB()

grid = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=10, scoring='accuracy', refit=True)
grid = grid.fit(X_train, y_train)

print('Best estimator:', grid.best_estimator_)
print('Best params:', grid.best_params_)
print('Best Score:', grid.best_score_)


# In[42]:


# Naive Bayes
nb = grid.best_estimator_
nb.fit(X_train, y_train)
print("Naive Bayes Results:")
nb_results = evaluate_model_performance(nb, X_test, y_test)


# # ANN

# In[43]:


pip install scikeras


# In[44]:


pip install tensorflow


# In[45]:


from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras import models
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def one_layer_model(num_of_input_features, hidden_layer_nodes=20):
    """ 
    We wrap the model into a function for better usability.
    We make some of the important hyper-parameters, like the learning rate, 
    as arguments to the function. This way we can test different values for these
    hyperparameters without having to change the hard-coded model itself.
    """
    # create a simple model with ONE hidden layer only
    model = models.Sequential()
    # we create a hidden layer with 20 nodes. 
    # Here we can directly give it the input shape. Otherwise we can also create a separate input layer
    model.add(layers.Dense(hidden_layer_nodes, input_dim=num_of_input_features, activation='relu'))
    model.add(layers.Dense(7, activation='softmax')) #is it clear why here we use "sigmoid" and use "softmax" for multi-class problems?
    return model


clf = KerasClassifier(
    model=one_layer_model,
    loss="sparse_categorical_crossentropy",
    model__hidden_layer_nodes=20,
    num_of_input_features= X_train.shape[1],
    epochs= 50,
    batch_size= 64,
    verbose= 0,
    validation_split= 0.2,
    optimizer = "adam",
    optimizer__learning_rate = 0.001,
)

params = {
    'optimizer__learning_rate': [0.01, 0.001],
    'model__hidden_layer_nodes': [20,30,40],
}

gs = GridSearchCV(clf, params, scoring='balanced_accuracy', verbose=True)

gs.fit(X_train, y_train)

print(gs.best_score_, gs.best_params_)


# In[46]:


# Check performance on test data
test_probabilities = gs.best_estimator_.predict(X_test)
# let's show the classification report with all the metrics
evaluate_model_performance(gs.best_estimator_, X_test, y_test)


# In[ ]:




