import requests
from os import path


def get_category_members(category):
    
    url = f'https://en.wiktionary.org/w/api.php?action=query&list=categorymembers&cmlimit=max&format=json&cmtitle={category}'
    response = requests.get(url)
    data = response.json()
    members = [x['title'] for x in data['query']['categorymembers'] if x['ns'] == 0]

    while 'continue' in data:
        new_url = url + '&cmcontinue=' + str(data['continue']['cmcontinue'])
        response = requests.get(new_url)
        data = response.json()
        members.extend([x['title'] for x in data['query']['categorymembers'] if x['ns'] == 0])

    return members


def main(category, members_filepath):

    if path.isfile(members_filepath):
        with open(members_filepath, 'r') as infile:
            members = [x.replace('\n', '') for x in infile.readlines()]
        print(f'Existing file contains {len(members)} entries')
    else:
        print(f'Retrieving members of "{category}"...')
        members = get_category_members(category)
        print(f'Saving {len(members)} entries...')
        with open(members_filepath, 'w') as outfile:
            for member in members:
                outfile.write(member+'\n')
    

if __name__ == '__main__':

    category = 'Category:English_blends'
    members_path = '../../Data/Wiktionary/blends.txt'
    main(category, members_path)
