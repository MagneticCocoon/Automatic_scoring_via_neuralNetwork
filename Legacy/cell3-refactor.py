# Text representation: vector
# Converting text to numbers

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

pd.concat([stdnt_answers, pd.DataFrame(sem_sims)], axis=1, sort=False).to_csv("semsim_field_max_vector.csv", na_rep="OOV", index = False)