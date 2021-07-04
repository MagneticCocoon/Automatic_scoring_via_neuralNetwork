from sklearn.neural_network import MLPClassifier
import pickle
import spacy

MODEL_FILE = 'pickled_nnmodel'

def LoadModelFromFile(filename: str):
    infile = open(filename, "rb")
    NN = pickle.load(infile)
    infile.close()
    return NN


if __name__ == "__main__":
    nl_spcy_md = spacy.load("nl_core_news_md")

    NN = LoadModelFromFile(MODEL_FILE)
    print(NN)

    A = 'Rode metrowagons tot zinken gebracht'
    sample = A.lower().replace("\r", " ").replace("\n", " ")    # These two already exist in TrainModel.py
    A_wvec = nl_spcy_md(sample).vector                          # Should refactor to a function

    # Reshape your data either using array.reshape(-1, 1) if your data has a single feature
    # or array.reshape(1, -1) if it contains a single sample.
    prediction = NN.predict(A_wvec.reshape(1, -1)) # works
    print(prediction)
