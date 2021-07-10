# Test the functions in SemanticSimilarity.py by going through each training answer
# and check the predicted slot against the verbandnummer.


import pandas as pd
import SemanticSimilarity

STUDENT_ANSWERS = "Data/trainingSet.csv"

testData = pd.read_csv(STUDENT_ANSWERS, sep=';')
testData = testData[ (testData["Code"] == "c") & (testData["Verbandnummer"].notnull()) & (testData["Verbandnummer"] != "0") ]

wrongPredictions = 0
totalPredictions = 0

if __name__ == "__main__":

    for index, row in testData.iterrows():
        if index % 500 == 0:
            print(index)

        try:
            studentAnswer   = SemanticSimilarity.TrimText(row["Veld"])
            verbandnummer   = int(row["Verbandnummer"])
            task            = row["Tekstnaam"].strip()  # Remove any spaces that might be there, jsut to be safe
        except:
            print("Could not parse: " + row)
            continue  # If any error occurse we just skip to the next row

        projection = SemanticSimilarity.nlp(studentAnswer)

        (mostSimilarAnswer, similarity) = SemanticSimilarity.FindMostSimilarModelAnswer(projection, task)
        if mostSimilarAnswer != verbandnummer:
            wrongPredictions += 1
        totalPredictions += 1

    print("Test complete! There were {total} tested answers with {error} incorrect ({percent:.4g}%).\n".format(
        total=totalPredictions, error=wrongPredictions, percent=100.0 * (wrongPredictions / totalPredictions)))