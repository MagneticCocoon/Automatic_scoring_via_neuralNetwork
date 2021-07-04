# Text representation: vector
# Converting text to numbers

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import re
import pandas as pd
import spacy

# READ INPUT files with student answers and model responses
INPUT_FILE = 'semsim_field_max02.csv'

# separator is a semicolon because some fields contain commas
stdnt_answers = pd.read_csv(INPUT_FILE, sep=',')

nl_spcy_md = spacy.load("nl_core_news_md")
wv_model = 'spacy_md'

# word vector models
sem_sims = {wv_model + "_vector": [], wv_model + "_norm": []}

for index, row_stdnt in stdnt_answers.iterrows():
    sample = row_stdnt['Veld'].lower().replace("\r"," ").replace("\n"," ")
    sem_sims['spacy_md_vector'].append(nl_spcy_md(sample).vector)
    sem_sims['spacy_md_norm'].append(nl_spcy_md(sample).vector_norm)



# (Binary) Classification: g, c

print("Binary classification")

sem_sim = pd.concat([stdnt_answers, pd.DataFrame(sem_sims)], axis=1, sort=False)

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

print("Preprocessing ...")
X = semsim_gc[wv_model + '_vector']
X = X.apply(pd.Series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_portion, shuffle=True, random_state=0)

print(X_train)

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NN.fit(X_train, y_train)
# NN.predict(X_test)

print(round(NN.score(X_test, y_test), 4))

pd.DataFrame(accuracies).to_csv("classification_accuracies.csv", index=False)
