import json
import pandas as pd
from wiktionaryparser import WiktionaryParser
from collections import defaultdict
from tqdm import tqdm

def get_entry_info(entry, print_definitions=False):
    
    parser = WiktionaryParser()
    parser.set_default_language('english')
    #parser.exclude_part_of_speech('verb')
    entry_info = parser.fetch(entry)
    
    entry_dic = defaultdict(list)

    for i, item in enumerate(entry_info):
        for definition in item['definitions']:
            pos = definition['partOfSpeech']
            texts = definition['text'][1:]
            examples = definition['examples']
            sense_dic = {'definitions': [], 'categories':[], 'examples': examples}
            for x, text in enumerate(texts):
                if text.startswith('('):
                    startid = text.index(')')
                    categories = text[1:startid].split(', ')
                    text = text[startid+2:]
                else:
                    categories = []
                sense_dic['definitions'].append(text)
                sense_dic['categories'].append(categories)

            entry_dic[pos].append(sense_dic)

    if print_definitions:
        
        for pos, senses in entry_dic.items():
            print(pos.upper())
            for i, sense in enumerate(senses):
                print(f'({i+1})')
                for x, (definition, categories) in enumerate(zip(sense['definitions'], sense['categories'])):
                    print(f'\t{x+1}. {definition} | Categories: {categories}')

    return entry_dic       


def entryinfo_json_to_senseinfo_csv(file):

    with open(file, 'r') as infile:
        content = json.load(infile)

    entries = []
    for term in content:
        ambiguous = len(content[term].keys()) > 1 # true when multiple pos-tags
        for pos in content[term]:
            i = 0
            if not ambiguous:
                ambiguous = len(content[term][pos]) > 1 # true when multiple etymologies
            for entry in content[term][pos]:
                defs = entry['definitions']
                cats = entry['categories']
                examples = entry['examples']
                for defo, cat in zip(defs, cats):
                    i += 1
                    if not ambiguous:
                        ambiguous = len(defs) > 1 # true when multiple definitions
    
                    entry = {'term': term,
                            'pos': pos,
                            'ambigous': ambiguous,
                            'sense_id': i,
                            'definition': defo,
                            'categories': ', '.join(cat), 
                            'n_examples': len(examples),
                            'examples': '\n'.join(examples)}
                    entries.append(entry)

    df = pd.DataFrame(entries)
    df.to_csv(file+'.csv', index=False)


def main(members_filepath, output_path):
    
    with open(members_filepath, 'r') as infile:
        content = infile.readlines()
    members = [m.replace('\n', '') for m in content]
        
    print(f'Retrieving information for all entries...')
    members_info = dict()
    for member in tqdm(members):
        info_dic = get_entry_info(member)
        members_info[member] = info_dic

    print(f'Saving output...')
    with open(output_path+'.json', 'w') as outfile:
        json.dump(members_info, outfile)

    entryinfo_json_to_senseinfo_csv(output_path)


if __name__ == '__main__':

    members_path = '../../data/wiktionary_hateterms_final.txt'
    output_path = '../../data/hateterms-entries' #filename without extension
    main(members_path, output_path)

    #entry = 'coconut'
    #get_entry_info(entry, print_definitions=True)