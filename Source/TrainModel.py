# Text representation: vector
# Converting text to numbers

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import re
import pandas as pd
import pickle
import spacy

INPUT_FILE  = 'semsim_field_max02.csv'
OUTPUT_FILE = 'pickled_nnmodel'

stdnt_answers = pd.read_csv(INPUT_FILE, sep=',', skipinitialspace=True)

# The highest accuracy is obtained by using Neural networks with spacy_md and a training size of 0.4
clas_algorithm  = 'Neural Networks'
wv_model = 'spacy_md'
nl_spcy_md = spacy.load("nl_core_news_md")
test_portion    = 0.4

# word vector models
sem_sims = {'word_vector': [], 'word_vector_norm': []}

for index, row_stdnt in stdnt_answers.iterrows():
    sample = row_stdnt['Veld'].lower().replace("\r"," ").replace("\n"," ")
    sem_sims['word_vector'].append(nl_spcy_md(sample).vector)
    sem_sims['word_vector_norm'].append(nl_spcy_md(sample).vector_norm)

# (Binary) Classification: g, c

print("Binary classification")

sem_sim = pd.concat([stdnt_answers, pd.DataFrame(sem_sims)], axis=1, sort=False)

semsim_gc = sem_sim[sem_sim.Code.isin(['g', 'c'])]

# Remove any whitespace that might exist in the cells
semsim_gc['Code'] = semsim_gc['Code'].map(str.strip)
semsim_gc['Tekstnaam'] = semsim_gc['Tekstnaam'].map(str.strip)

semsim_gc['Code'] = semsim_gc['Code'].map({'g': 1, 'c': 0})
semsim_gc['Tekstnaam'] = semsim_gc['Tekstnaam'].map({'Metro': 1, 'Botox': 2, 'Suez': 3, 'Muziek': 4, 'Geld': 5, 'Beton': 6})

# Preprocessing for classification
print("Preprocessing ...")

y = semsim_gc.Code
X = semsim_gc['word_vector']
X = X.apply(pd.Series)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_portion, shuffle=True, random_state=0)
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NN.fit(X_train, y_train)
# NN.predict(X_test)

print("Model trained with accuracy %f" % round(NN.score(X_test, y_test), 4))

# Dump to file
outfile = open(OUTPUT_FILE, "wb")
pickle.dump(NN, outfile)
outfile.close()
print("Saved model to file: %s" % OUTPUT_FILE)
