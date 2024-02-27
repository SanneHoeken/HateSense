import json
import pandas as pd

def main(senseinfo_file, group2cat_file=None):

    df = pd.read_csv(senseinfo_file, sep=';')
    if group2cat_file:
        group2cat = json.load(group2cat_file)
    examples = []
    i = 0
    
    for row in df.iterrows():
        row = row[1]
        if type(row['examples']) == str:
            row_examples = row['examples'].split('\n')
            for ex in row_examples:
                i += 1
                ex_entry = {'id': i,
                            'example': ex,
                            'term': row['term'],
                            'pos': row['pos'],
                            'sense_id': row['sense_id'],
                            'definition': row['definition'],
                            'categories': row['categories'],
                            'hate': row['hate']} # TODO: remove after group2cat file is applied
               
                # GROUP CATEGORY COLUMNS
                if group2cat_file:
                    cats = []
                    if type(row['categories']) == str:
                        cats = row['categories'].split(',')
                    for group_label, group in group2cat:
                        group_cats = []
                        for cat in cats:
                            if cat in group:
                                group_cats.append(cat)
                        ex_entry[group_label] = ', '.join(group_cats)
                
                examples.append(ex_entry)
    
    new_df = pd.DataFrame(examples)
    new_df.to_csv(file.replace('.csv', '-examples.csv'), index=False)
    
if __name__ == '__main__':

    file = '../../data/hateterms-senses.csv'
    #group2cat_file = '../../data/group2cat.json'
    
    main(file)

    """TODO: fix incorrect new lines in resulting file"""