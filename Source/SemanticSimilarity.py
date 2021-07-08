
import pandas as pd
import spacy

def TrimText(txt: str):
    """ Converts a string to lower case and removes new lines. """
    return txt.lower().replace("\r", " ").replace("\n", " ")


def get_similarity(sample_doc, model_doc):
    """ Returns the semantic similarity between two documents that have been processed by the spacy pipeline. """
    sem_sim = None
    oov = None
    if sample_doc.vector_norm > 0:
        if model_doc.vector_norm > 0:
            sem_sim = sample_doc.similarity(model_doc)
        else:
            oov = model_doc
    else:
        oov = sample_doc

    return sem_sim, oov


MODEL_FILE = 'model_answers.csv'

nlp = spacy.load("nl_core_news_md")

# These will be the parameters when implemented on the server
# Student id: 120, Text: Muziek
Task  = "Muziek"
Slot1 = "lezen van muziek-noten"                                # g, Should be: 1
Slot2 = None
Slot3 = "12-jarige hoger IQ"                                    # g, Should be: 5
Slot4 = "betere vaardigheden als wiskundige breuken berekenen"  # g, Should be: 3
Slot5 = "ophalen van herinneringen"                             # g, SHould be: 4

# Load model answers and throw away those we don't need
modelAnswers = pd.read_csv(MODEL_FILE, sep=';', index_col='TextName')
modelAnswers = modelAnswers.loc[Task]

# Dictionary mapping slot number to a pair: ("Most similar model answer", "Similarity")
MostSimilarModelAnswer = { 1: (None, None), 2: (None, None), 3: (None, None), 4: (None, None), 5: (None, None) }

# Pack into array, preprocess and run throught the spacy pipeline
Slots = [Slot1, Slot2, Slot3, Slot4, Slot5]
Slots = list(map(lambda slotText: TrimText(slotText) if slotText is not None else slotText, Slots))

for slotIndex, text in Slots:
    # Process the student answer
    studentAnswerProjection = nlp(text)

    for modelAnswerIndex, modelAnswer in modelAnswers:
        modelAnswer = TrimText(modelAnswer)
        modelAnswerProjection = nlp(modelAnswer)

        sem_sim, oov = get_similarity(studentAnswerProjection, modelAnswerProjection)
        if sem_sim is not None:
            slotNumber = slotIndex + 1
            modelAnswerNumber = modelAnswerIndex + 1

            (field, similarity) = MostSimilarModelAnswer[slotNumber]
            if field is None or similarity is None:
                MostSimilarModelAnswer[slotNumber] = (modelAnswerNumber, sem_sim)
            elif sem_sim > MostSimilarModelAnswer[slotNumber]:
                MostSimilarModelAnswer[slotNumber] = sem_sim if sem_sim <= 1 else 1
            #else:  # TODO: Figure out what to do with this case
                #OOV_list.append(oov)
                #sem_sim_2save = 'OOV'  # out of vocabulary


print()

