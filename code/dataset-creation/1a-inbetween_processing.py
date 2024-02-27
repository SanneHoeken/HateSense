
file1 = '../Data/Wiktionary/wiktionary_derogatory-terms.txt'
file2 = '../Data/Wiktionary/wiktionary_offensive-terms.txt'
file3 = '../Data/Wiktionary/wiktionary_people.txt'

# DEROGATORY
with open(file1, 'r') as infile:
    elements1 = [x.replace('\n', '') for x in infile.readlines()]

# OFFENSIVE
with open(file2, 'r') as infile:
    elements2 = [x.replace('\n', '') for x in infile.readlines()]

# PEOPLE
with open(file3, 'r') as infile:
    elements3 = [x.replace('\n', '') for x in infile.readlines()]

# DEROGATORY or OFFENSIVE
union12 = set(elements1).union(set(elements2))

# (DEROGATORY or OFFENSIVE) and PEOPLE
intersect3 = union12.intersection(set(elements3))

with open('../../data/wiktionary_hateterms_final.txt', 'w') as outfile:
    for w in intersect3:
        outfile.write(w+'\n')


"""
file4 = '../Data/Wiktionary/wiktionary_blends.txt'

# BLENDS
with open(file4, 'r') as infile:
    elements4 = [x.replace('\n', '') for x in infile.readlines()]

# (DEROGATORY or OFFENSIVE) and BLENDS
intersect4 = union12.intersection(set(elements4))

# (DEROGATORY or OFFENSIVE) and PEOPLE and BLENDS
intersect34 = intersect3.intersection(set(elements4))

with open('../Data/Wiktionary/blends&people&offensive-derogatory.txt', 'w') as outfile:
    for w in intersect34:
        outfile.write(w+'\n')
"""