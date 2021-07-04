# (Binary) Classification: g, c

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
import re

INPUT_FILE = 'semsim_field_max_vector.csv'

sem_sim = pd.read_csv(INPUT_FILE, sep=',', na_values="OOV")  # , nrows=15, skiprows=range(1,725))
semsim_gc = sem_sim[sem_sim.Code.isin(['g', 'c'])]
semsim_gc['Code'] = semsim_gc['Code'].str.replace('g', '1')
semsim_gc['Code'] = semsim_gc['Code'].str.replace('c', '0')
semsim_gc['Tekstnaam'] = semsim_gc['Tekstnaam'].str.replace('Metro', '1').str.replace('Botox', '2'). \
    str.replace('Suez', '3').str.replace('Muziek', '4').str.replace('Geld', '5').str.replace('Beton', '6')

# The highest accuracy is obtained by using Neural networks with spacy_md and a
# training size of 0.4
wv_model        = 'spacy_md'
clas_algorithm  = 'Neural Networks'
test_portion    = 0.4

accuracies = {'algorithm': [], 'test_size': []}
y = semsim_gc.Code
# aux = semsim_gc

# Preprocessing for classification
# Changing from a vector of dimension 300 to 300 columns for each (Dutch) language model
accuracies[wv_model] = []
#semsim_gc[wv_model + '_vector'] = semsim_gc[wv_model + '_vector'].apply(lambda x: re.sub(r"\s+", " ", x)). \
#    str.replace('[', '').str.replace(']', '').str.strip()

semsim_gc[wv_model + '_vector'] = semsim_gc[wv_model + '_vector'].apply(lambda x: re.sub(r"\s+", " ", x))
semsim_gc[wv_model + '_vector'] = semsim_gc[wv_model + '_vector'].apply(lambda x: re.sub(r"\[", "", x))
semsim_gc[wv_model + '_vector'] = semsim_gc[wv_model + '_vector'].apply(lambda x: re.sub(r"\]", "", x))
semsim_gc[wv_model + '_vector'] = semsim_gc[wv_model + '_vector'].str.strip()

X = semsim_gc[wv_model + '_vector'].str.split(expand=True, )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_portion, shuffle=True, random_state=0)

print(X_train)

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NN.fit(X_train, y_train)
print(round(NN.score(X_test, y_test), 4))
# NN.predict(X_test)

pd.DataFrame(accuracies).to_csv("classification_accuracies.csv", index=False)
