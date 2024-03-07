import pickle, json
import pandas as pd
from sklearn.model_selection import KFold
from mlp import train_test_MLP

def main(data_path, label_column, label_encoder, embedding_path, output_path):

    random_state = 0
    num_classes = len(set(label_encoder.values()))

    # load data
    with open(embedding_path, 'rb') as infile:
        embeddings = pickle.load(infile)

    data = pd.read_csv(data_path).sample(frac=1, random_state=3, ignore_index=True)
    data[label_column] = data[label_column].fillna('NaN')
    data['encoded_label'] = data[label_column].replace(label_encoder)
    data = data[data['id'].isin(embeddings)]

    # intialize 5-fold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    kfold_accuracies = []
    test_rows = []
    test_predictions = []

    unique_terms = data['term'].unique()
    for i, (train_indices, test_indices) in enumerate(kf.split(unique_terms)):
        
        # get test and train data
        train_terms = [term for i, term in enumerate(unique_terms) if i in train_indices]
        test_terms = [term for i, term in enumerate(unique_terms) if i in test_indices]

        train_embeddings = []
        train_labels = []
        test_embeddings = []
        test_labels = []

        for _, row in data.iterrows():
            if row['term'] in train_terms:
                train_embeddings.append(embeddings[row['id']])
                train_labels.append(row['encoded_label'])
            elif row['term'] in test_terms:
                test_embeddings.append(embeddings[row['id']])
                test_labels.append(row['encoded_label'])
                row['test_fold_id'] = i + 1
                test_rows.append(row)

        # train and test MLP model
        predictions, accuracy = train_test_MLP(train_embeddings, train_labels, test_embeddings, test_labels, num_classes)
        kfold_accuracies.append(accuracy)
        test_predictions.extend(predictions)
        
        # print output
        #print(f"Fold {i+1}")
        #print(f"Train size: {len(train_labels)} / Test size: {len(test_labels)}")
        #print(f"Accuracy: {accuracy}\n")

    print('Avg accuracy:', sum(kfold_accuracies)/len(kfold_accuracies))
    
    # save preds
    output_df = pd.DataFrame(test_rows)
    output_df['predictions'] = test_predictions
    output_df.to_csv(output_path)


if __name__ == "__main__":

    output_path = '../../output/mlp_bertbase_lastlayer_labels1.csv'
    embedding_path = '../../output/id2bertbase-lastlayer'
    data_path = '../../data/hateterms-senses-examples.csv' 
    label_column = 'hate'
    label_encoder_file = '../../data/hatelabel_encoder.json'
    
    with open(label_encoder_file, 'r') as infile:
        label_encoder = json.load(infile)

    main(data_path, label_column, label_encoder, embedding_path, output_path)