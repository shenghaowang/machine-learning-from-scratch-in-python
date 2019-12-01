### Frequent itemsets generation
def generate_candidates(itemsets):
    """Generate new frequent k-itemsets based on the frequent (k - 1)-itemsets
    Args:
        itemsets (list): frequent (k - 1)-itemsets generated from the prior iteration

    Returns:
        k-itemsets (list): new frequent k-itemsets
    """
    k_itemsets = []
    if itemsets:
        N = len(itemsets)
        for i in range(N-1):
            for j in range(i+1, N):
                item_list1 = sorted(itemsets[i])
                item_list2 = sorted(itemsets[j])
                if item_list1[:-1] == item_list2[:-1]:
                    k_itemsets.append(itemsets[i] | itemsets[j])
    return k_itemsets


def prune_candidates(candidates, freq_itemsets):
    """Eliminate k-itemsets whose subsets are known to be infrequent in previous iteration
    Args:
        candidates (list): frequent k-itemsets generated based on the frequent (k - 1)-itemsets
        frequent_itemsets (list): frequent (k - 1)-itemsets generated from the last iteration

    Returns:
        pruned_candidates (list): pruned candidate k-itemsets
    """
    pruned_candidates = []
    if candidates:
        for candidate in candidates:
            to_be_pruned = False
            for item in candidate:
                candidate_copy = set(candidate.copy())
                candidate_copy.remove(item)
                if candidate_copy not in freq_itemsets:
                    to_be_pruned = True
                    break
            if not to_be_pruned:
                pruned_candidates.append(candidate)

    return pruned_candidates


def count_freq(transactions, candidates):
    """Count the number of occurrences of the candidate k-itemsets
    Args:
        transactions (list): a list of transaction record with items involved in the transaction
        candidates (list): candidate frequent k-itemsets obtained from candidate generation and pruning

    Returns:
        item_freq (dict): a dictionary which stores the candidate frequent itemsets and their correcsponding number of occurrences
    """
    item_freq = {}
    for candidate in candidates:
        for transaction in transactions:
            if candidate.issubset(frozenset(transaction)):
                if candidate in item_freq:
                    item_freq[candidate] += 1
                else:
                    item_freq[candidate] = 1
    return item_freq


### Association rule generation
def prune_consequents(fk, H, support_counts_dict, minconf):
    """Generate association rules and eliminate the consequent item which results in low confidence
    Args:
        fk (list): a frequent itemset
        H (list): a list of item combinations to be used as consequence
        support_counts_dict (dict): a dictionary which stores the frequent itemsets and their respective frequency
        minconf (float): minimum acceptable confidence of association rules

    Returns:
        rules (list): generated association rules

    """
    rules = []
    low_conf_consequents = []
    for h in H:
        fk_list = list(fk.copy())
        fk_list.remove(list(h)[0])
        antecedent = frozenset(fk_list)
        conf = support_counts_dict[fk] / support_counts_dict[antecedent]
        if conf >= minconf:
            new_rule = list(antecedent)
            new_rule.append('=>')
            new_rule.extend(list(h))
            rules.append(new_rule)
        else:
            low_conf_consequents.append(h)
    H = [h for h in H if h not in low_conf_consequents]
    return rules, H


def ap_genrules(fk, Hm, support_counts_dict, minconf):
    """Generate association rules with more than 1 items as consequent
    Args:
        fk (list): a frequent itemset
        Hm (list): a list of item combinations to be used as consequence
        support_counts_dict (dict): a dictionary which stores the frequent itemsets and their respective frequency
        minconf (float): minimum acceptable confidence of association rules

    Returns:
        assoc_rules (list): generated association rules
    """
    m = 1
    assoc_rules = []
    while len(fk) > m + 1:
        Hmp1 = generate_candidates(Hm)
        Hmp1 = prune_candidates(Hmp1, Hm)
        rules, Hm = prune_consequents(fk, Hmp1, support_counts_dict, minconf)
        assoc_rules.extend(rules)
        m += 1
    return assoc_rules
