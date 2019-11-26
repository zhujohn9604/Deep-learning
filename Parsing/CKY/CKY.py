RULE_DICT = {}


def read_grammar(grammar_file):
    """
    Reads in the given grammar file and splits it into separate lists for each rule.
    grammar_file: the grammar file to read in.
    return: the list of rules.
    """
    with open(grammar_file) as cfg:
        lines = cfg.readlines()
    return [line.replace('->', '').split() for line in lines]


def add_rule(rule):
    """
    Adds a rule to the dictionary of lists of rules.
    :param rule: the rule to add to the dict.
    """
    global RULE_DICT

    if rule[0] not in RULE_DICT:
        RULE_DICT[rule[0]] = []
    RULE_DICT[rule[0]].append(rule[1:])


def convert_grammar(grammar):
    """
    Convert a context-free grammar in the form of

    S -> NP VP
    NP -> Det ADV N
    and so on into a chomsky normal form (CNF) of that grammar.
    After the conversion
    :param grammar:
    :return:
    """
    global RULE_DICT
    unit_productions, result = [], []
    result_append = result.append
    index = 0

    for rule in grammar:
        new_rules = []
        if len(rule) == 2 and rule[1][0] != "'":
            # unit production : A -> X
            unit_productions.append(rule)
            # add_rule(rule)
            continue
        elif len(rule) > 2:
            # Rule is in form A -> B C D or A -> B c
            terminals = [(i, item) for i, item in enumerate(rule) if item[0] == "'"]
            if terminals:
                # if terminals exist on RHS: A -> B c
                # Change A -> B c to A -> B A1 (dummy); A1 -> c
                for terminal in terminals:
                    # convert terminals within rules to dummy non-terminals
                    rule[terminal[0]] = f'{rule[0]}{str(index)}'
                    new_rules += [f'{rule[0]}{str(index)}'] + [terminal[1]]
                index += 1
            # All rule in A -> B C ...
            while len(rule) > 3:
                # example: A -> B C D E => A -> A1 D E => A -> A2 E
                new_rules += [f'{rule[0]}{str(index)}']+ [rule[1]] + [rule[2]]
                rule = [rule[0]] + [f'{rule[0]}{str(index)}'] + rule[3:]
                index += 1
        add_rule(rule)
        result_append(rule)  # results of current rule
        if new_rules:
            result_append(new_rules)

    # Handle the unit productions
    while unit_productions:
        rule = unit_productions.pop()  # save and delete the last one from the lists
        if rule[1] in RULE_DICT:
            for item in RULE_DICT[rule[1]]:
                new_rule = [rule[0]] + item
                if len(new_rule) > 2 or new_rule[1][0] == "'":
                    result_append(new_rule)
                else:
                    unit_productions.append(new_rule)
                add_rule(new_rule)
    return result










