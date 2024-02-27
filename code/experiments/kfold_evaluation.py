import pickle
import pandas as pd
from sklearn.model_selection import KFold
from mlp import train_test_MLP

def main(data_path, embedding_path, label_column, label_encoder):

    # load data
    data = pd.read_csv(data_path)
    unique_terms = data['term'].unique()

    with open(embedding_path, 'rb') as infile:
        embeddings = pickle.load(infile)

    # intialize 5-fold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    accuracies = []
    for i, (train_indices, test_indices) in enumerate(kf.split(unique_terms)):
        
        # get test and train data
        train_terms = [term for i, term in enumerate(unique_terms) if i in train_indices]
        test_terms = [term for i, term in enumerate(unique_terms) if i in test_indices]

        train_embeddings = []
        train_labels = []
        test_embeddings = []
        test_labels = []

        for row in data.iterrows():
            row = row[1]
            if row['id'] in embeddings:
                if row['term'] in train_terms:
                    train_embeddings.append(embeddings[row['id']])
                    train_labels.append(label_encoder[row[label_column]])
                elif row['term'] in test_terms:
                    test_embeddings.append(embeddings[row['id']])
                    test_labels.append(label_encoder[row[label_column]])

        # train and test MLP model
        accuracy = train_test_MLP(train_embeddings, train_labels, test_embeddings, test_labels)
        accuracies.append(accuracy)
        
        # print output
        print(f"Fold {i+1}")
        print(f"Train size: {len(train_labels)} / Test size: {len(test_labels)}")
        print(f"Accuracy: {accuracy}\n")

    print('Avg accuracy:', sum(accuracies)/len(accuracies))
    

if __name__ == "__main__":

    embedding_path = '../../output/id2bertbase-lastlayer'
    data_path = '../../data/hateterms-senses-examples.csv' 
    label_column = 'hate'
    label_encoder = {True: 1, False: 0}

    main(data_path, embedding_path, label_column, label_encoder)