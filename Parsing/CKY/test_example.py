from CKY import *

grammar = read_grammar('L1_mini_grammar.txt')

CKY_results = convert_grammar(grammar)

f = open('L1_CNF.txt', 'w')
for rule in CKY_results:
    if len(rule) == 2:
        f.write(str(rule[0]) + ' -> ' + str(rule[1]) + '\n')
    else:
        f.write(str(rule[0]) + ' -> ' + str(rule[1]) + ' ' + str(rule[2]) + '\n')


f.close()

