import torch, pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def extract_embedding(model, example_encoding, term_indices, layers):

    # feed example encodings to the model    
    input_ids = torch.tensor([example_encoding])
    encoded_layers = model(input_ids)[-1]
    
    # extract selection of hidden layer(s)
    if type(layers) == int:
        vecs = encoded_layers[layers].squeeze(0)
    elif type(layers) == list:
        selected_encoded_layers = [encoded_layers[x] for x in layers]
        vecs = torch.mean(torch.stack(selected_encoded_layers), 0).squeeze(0)
    elif layers == 'all':
        vecs = torch.mean(torch.stack(encoded_layers), 0).squeeze(0)
    
    # target word selection 
    vecs = vecs.detach()
    start_idx, end_idx = term_indices
    vecs = vecs[start_idx:end_idx]
    
    # aggregate sub-word embeddings (by averaging)
    vector = torch.mean(vecs, 0)
    
    return vector


def main(input_path, output_path, model_name, layers='all'):

    data = pd.read_csv(input_path)
    tknzr = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    # TO BE FIXED! No exact match between term in 'term' column and mention of term in example, resulting in term_indices = None for 1276 out of 4671 cases!
    non_exact_matches = 0 #TMP

    embeddings = dict()

    for row in tqdm(data.iterrows()):
        row = row[1]
        id = row['id']
        term = row['term']
        example = row['example'] 

        # encode example
        example_encoding = tknzr.encode(example.lower(), truncation=True) #for batched processing: set padding='max_length'
        term_encoding = tknzr.encode(term, add_special_tokens=False)

        # find indices for target term
        term_indices = None
        for i in range(len(example_encoding)):
            if example_encoding[i:i+len(term_encoding)] == term_encoding:
                term_indices = (i, i+len(term_encoding))

        # extract embedding
        if term_indices:
            vector = extract_embedding(model, example_encoding, term_indices, layers=layers)
            embeddings[id] = vector

        else: #TMP
            non_exact_matches += 1
    
    print("Number of non exact matches:", non_exact_matches) #TMP

    with open(output_path, 'wb') as outfile:
        pickle.dump(embeddings, outfile)
        

if __name__ == '__main__':
    
    input_path = '../../data/hateterms-senses-examples.csv' 
    output_path = '../../output/id2bertbase-lastlayer'
    model_name = 'bert-base-uncased'
    
    main(input_path, output_path, model_name, layers=-1)