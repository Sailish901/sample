import imblearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier  
import numpy as np
import pandas as pd




url = 'https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv'
data = pd.read_csv(url)


X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


z = 1.97 
m = 0.06 

# Calculate the sample size for each sampling technique using the formula
n1 = int(np.ceil((z**2 * 0.5 * 0.5) / (m**2)))
n2 = int(np.ceil((z**2 * 0.05 * (1-0.05)) / (m**2)))
n3 = int(np.ceil((z**2 * 0.05 * (1-0.05)) / (m**2)))
n4 = int(np.ceil((z**2 * 0.05 * (1-0.05)) / (m**2)))
n5 = int(np.ceil((z**2 * 0.05 * (1-0.05)) / (m**2)))

# Define the sampling techniques and models
sample_1 = RandomUnderSampler(sampling_strategy='majority', random_state=56)
sample_2 = RandomOverSampler(sampling_strategy='minority', random_state=98)
sample_3 = SMOTE(sampling_strategy='minority', random_state=95)
sample_4 = TomekLinks(sampling_strategy='majority')
sample_5 = NearMiss(version=3, n_neighbors=3)


m1 = LogisticRegression(random_state=98,max_iter=1000)
m2 = DecisionTreeClassifier(random_state=50)
m3 = RandomForestClassifier(random_state=87)
m4 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
m5 = SGDClassifier(random_state=42, loss='modified_huber', max_iter=1000)


samplers = {
    'Sampling1': sample_1,
    'Sampling2': sample_2,
    'Sampling3': sample_3,
    'Sampling4': sample_4,
    'Sampling5': sample_5,
}
models = {
    'M1': m1,
    'M2': m2,
    'M3': m3,
    'M4': m4,
    'M5': m5,
}


results = {}
for sampler_name, sampler in samplers.items():
    if sampler_name == 'Sampling1':
        n = n1
    elif sampler_name == 'Sampling2':
        n = n2
    elif sampler_name == 'Sampling3':
        n = n3
    elif sampler_name == 'Sampling4':
        n = n4
    else:
        n = n5

    # Undersample or oversample the training data
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    # Limit the resampled data to the sample size
    if len(X_resampled) > n:
        X_resampled = X_resampled[:n]
        y_resampled = y_resampled[:n]
    
    for model_name, model in models.items():
        # Train the model on the resampled data
        model.fit(X_resampled, y_resampled)
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Calculate the accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        
        # Add the accuracy score to the results dictionary
        if model_name in results:
            results[model_name][sampler_name] = accuracy
        else:
            results[model_name] = {sampler_name: accuracy}
            
# Print the results
print('Results:')
print('        Sampling1   Sampling2   Sampling3   Sampling4   Sampling5')
for model_name, model_results in results.items():
    print(model_name, end='')
    for sampler_name in samplers.keys():
        if sampler_name in model_results:
            print(f'    {model_results[sampler_name]:.4f}   ', end='')
        else:
            print('              ', end='')
    print() 



