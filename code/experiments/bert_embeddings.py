import torch, pickle, re
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from difflib import get_close_matches

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


def find_target_indices(tknzr, example, term):
            
    # encode example and target term
    example_encoding = tknzr.encode(example, truncation=True)
    term_encoding = tknzr.encode(term, add_special_tokens=False)
    
    # find indices for target term
    term_indices = None
    for i in range(len(example_encoding)):
        if example_encoding[i:i+len(term_encoding)] == term_encoding:
            term_indices = (i, i+len(term_encoding))
    
    if not term_indices:
        new_term = None
        new_example = None
        
        # try plural (simple rules)
        if term + 's' in example:
            new_term = term + 's'
        elif term.replace('y', 'ies') in example:
            new_term = term.replace('y', 'ies')
        elif term.replace('man', 'men') in example:
            new_term = term.replace('man', 'men')
        else:
            # try to find the most similar word in the example
            potential_target = get_close_matches(term, example.split(), n=1, cutoff=0.6)
            if len(potential_target) == 1:
                most_similar = re.sub(r'[^\w\s-]','', potential_target[0])
                # replace the most similar word (for which we assume misspelling) with the target term
                new_example = example.replace(most_similar, term)
        
        if new_term or new_example:
            # encode new term or example
            if new_term:
                term_encoding = tknzr.encode(new_term, add_special_tokens=False)
            elif new_example:
                example_encoding = tknzr.encode(new_example, truncation=True)
            # try finding indices again
            for i in range(len(example_encoding)):
                if example_encoding[i:i+len(term_encoding)] == term_encoding:
                    term_indices = (i, i+len(term_encoding))
    
    return term_indices


def main(input_path, output_path, model_name, layers='all'):

    data = pd.read_csv(input_path)
    tknzr = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    non_exact_matches = [] 
    embeddings = dict()
    
    for row in tqdm(data.iterrows()):
        row = row[1]
        id = row['id']
        term = row['term'].lower()
        example = row['example'].lower() 

        term_indices = find_target_indices(tknzr, example, term)     

        if term_indices:
            # extract embedding
            example_encoding = tknzr.encode(example, truncation=True)
            vector = extract_embedding(model, example_encoding, term_indices, layers=layers)
            embeddings[id] = vector
            pass 
        else:
            non_exact_matches.append(id)
    
    print("Number of examples without target term matches (and therefore excluded):", len(non_exact_matches))
    
    with open(output_path, 'wb') as outfile:
        pickle.dump(embeddings, outfile)
        

if __name__ == '__main__':
    
    input_path = '../../data/hateterms-senses-examples_final.csv' 
    output_path = '../../output/id2bertbase-lastlayer'
    model_name = 'bert-base-uncased'
    
    main(input_path, output_path, model_name, layers=-1)