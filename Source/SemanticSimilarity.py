
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


MODEL_FILE = 'Data/model_answers.csv'

nlp = spacy.load("nl_core_news_md")

# These will be the parameters when implemented on the server
# Student id: 120, Text: Muziek
Task  = "Muziek"
Slot1 = "lezen van muziek-noten"                                # g, Should be: 1
Slot2 = None
Slot3 = "12-jarige hoger IQ"                                    # g, Should be: 5
Slot4 = "betere vaardigheden als wiskundige breuken berekenen"  # g, Should be: 3
Slot5 = "ophalen van herinneringen"                             # g, Should be: 4

# Load model answers and throw away those we don't need
modelAnswers = pd.read_csv(MODEL_FILE, sep=';', index_col='TextName')
modelAnswers = modelAnswers.loc[Task]

# Dictionary mapping slot number to a pair: ("Most similar model answer", "Similarity")
MostSimilarModelAnswer = { 1: (None, None), 2: (None, None), 3: (None, None), 4: (None, None), 5: (None, None) }

# Pack into array, preprocess and run throught the spacy pipeline
Slots = [Slot1, Slot2, Slot3, Slot4, Slot5]
Slots = list(map(lambda slotText: TrimText(slotText) if slotText is not None else slotText, Slots))


def FindMostSimilarModelAnswer(studentAnswerNlp):
    """ Given a student answer, processed by spacy, find the most similar model answer.
        Output: tuple of (modelAnswer: integer, similarity: number) """
    MostSimilarAnswer = None
    Similarity = 0

    modelAnswerIndex = 0
    for modelAnswer in modelAnswers:
        modelAnswer = TrimText(modelAnswer)
        modelAnswerProjection = nlp(modelAnswer)

        sem_sim, oov = get_similarity(studentAnswerProjection, modelAnswerProjection)
        if sem_sim is not None:
            slotNumber = slotIndex + 1
            modelAnswerNumber = modelAnswerIndex + 1

            (field, similarity) = (MostSimilarAnswer, Similarity)
            if field is None or similarity is None:
                (MostSimilarAnswer, Similarity) = (modelAnswerNumber, sem_sim)
            elif sem_sim > similarity:
                (MostSimilarAnswer, Similarity) = (modelAnswerNumber, sem_sim if sem_sim <= 1 else 1)
            # else:  # TODO: Figure out what to do with this case
            # OOV_list.append(oov)
            # sem_sim_2save = 'OOV'  # out of vocabulary

        modelAnswerIndex += 1

    return (MostSimilarAnswer, Similarity)


slotIndex = 0
for text in Slots:
    if text is not None:
        studentAnswerProjection = nlp(text)  # Process the student answer
        MostSimilarModelAnswer[slotIndex + 1] = FindMostSimilarModelAnswer(studentAnswerProjection)
    slotIndex += 1

print(MostSimilarModelAnswer)