import json
import pandas as pd

def main(senseinfo_file, group2cat_file=None):

    df = pd.read_csv(senseinfo_file)
    
    if group2cat_file:
        with open(group2cat_file, 'r') as infile:
            group2cat = json.load(infile)
    
    examples = []
    n = 0
    
    for i, row in df.iterrows():
        if type(row['examples']) == str:
            row_examples = row['examples'].split('\n')
            for ex in row_examples:
                n += 1
                ex_entry = {'id': n,
                            'example': ex,
                            'term': row['term'],
                            'pos': row['pos'],
                            'sense_id': row['sense_id'],
                            'definition': row['definition'],
                            'categories': row['categories']} 
               
                # GROUP CATEGORY COLUMNS
                if group2cat_file:
                    cats = []
                    if type(row['categories']) == str:
                        cats = [c.replace('"', "'") for c in row['categories'].split(', ')]
                    for group_label, group in group2cat.items():
                        group_cats = []
                        for cat in cats:
                            if cat in group:
                                group_cats.append(cat)
                        ex_entry[group_label] = ','.join(group_cats) 
                        
                examples.append(ex_entry)
    
    new_df = pd.DataFrame(examples)
    new_df.to_csv(file.replace('.csv', '-examples.csv'), index=False)
    
if __name__ == '__main__':

    file = '../../data/hateterms-senses.csv'
    group2cat_file = '../../data/group2cat.json'
    
    main(file, group2cat_file=group2cat_file)