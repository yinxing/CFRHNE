import config
import utils
import yelp_evaluation
import dblp_evaluation
import aminer_evaluation
import acm_evaluation

def evaluate_business_cluster(Y, emb):
    score = Y.evaluate_business_cluster(emb)

    print("___________________cluster_______________________")
    print(' NMI score = %.4f' % (score))
    return score

def evaluate_business_classification(Y, emb):

    micro_f1, macro_f1 = Y.evaluate_business_classification(emb)
    print("_________________clasfiction______________________")
    print(' micro_f1 = %.4f  micro_f1 = %.4f' % (micro_f1, macro_f1))
    return micro_f1, macro_f1

def evaluate_yelp_link_prediction(Y, emb):

    auc, f1, acc = Y.evaluation_link_prediction(emb)
    print("_________________link_prediction______________________")
    print(' auc = %.4f  f1 = %.4f acc = %.4f' % (auc, f1, acc))
    return f1, acc, auc

# -----------dblp------------

def dblp_cluster(Y, emb):
    score = Y.evaluate_author_cluster(emb)

    print("___________________cluster_______________________")
    print(' NMI score = %.4f' % (score))
    return score

def dblp_classification(Y, emb):

    micro_f1, macro_f1 = Y.evaluate_author_classification(emb)
    print("_________________clasfiction______________________")
    print(' micro_f1 = %.4f  micro_f1 = %.4f' % (micro_f1, macro_f1))
    return micro_f1, macro_f1

def dblp_link_prediction(Y, emb):


    auc, f1, acc = Y.evaluation_link_prediction(emb)
    print("_________________link_prediction______________________")
    print(' auc = %.4f  f1 = %.4f acc = %.4f' % (auc, f1, acc))
    return f1, acc, auc

# -------aminer---------

def aminer_cluster(Y, emb):
    score = Y.evaluate_aminer_cluster(emb)

    print("___________________cluster_______________________")
    print(' NMI score = %.4f' % (score))
    return score

def aminer_classification(Y, emb):

    micro_f1, macro_f1 = Y.evaluate_aminer_classification(emb)
    print("_________________clasfiction______________________")
    print(' micro_f1 = %.4f  micro_f1 = %.4f' % (micro_f1, macro_f1))
    return micro_f1, macro_f1

def aminer_link_prediction(Y, emb):


    auc, f1, acc = Y.aminer_link_prediction(emb)
    print("_________________link_prediction______________________")
    print(' auc = %.4f  f1 = %.4f acc = %.4f' % (auc, f1, acc))
    return f1, acc, auc
# --------acm-----------

def acm_cluster(Y, emb):
    score = Y.evaluate_acm_cluster(emb)

    print("___________________cluster_______________________")
    print(' NMI score = %.4f' % (score))
    return score

def acm_classification(Y, emb):

    micro_f1, macro_f1 = Y.evaluate_acm_classification(emb)
    print("_________________clasfiction______________________")
    print(' micro_f1 = %.4f  micro_f1 = %.4f' % (micro_f1, macro_f1))
    return micro_f1, macro_f1

def acm_link_prediction(Y, emb):


    auc, f1, acc = Y.acm_link_prediction(emb)
    print("_________________link_prediction______________________")
    print(' auc = %.4f  f1 = %.4f acc = %.4f' % (auc, f1, acc))
    return f1, acc, auc



def all_evaluation(emb):
    dataset = config.dataset
    if(dataset == "dblp"):
        Y = dblp_evaluation.DBLP_evaluation()
        f1, acc, auc = dblp_link_prediction(Y, emb)
        nmi = dblp_cluster(Y, emb)
        mi_f1, ma_f1 = dblp_classification(Y, emb)

        return f1, acc, auc, nmi, mi_f1, ma_f1

    elif(dataset == "yelp"):
        Y = yelp_evaluation.Yelp_evaluation()
        f1, acc, auc = evaluate_yelp_link_prediction(Y, emb)
        nmi = evaluate_business_cluster(Y, emb)
        mi_f1, ma_f1 = evaluate_business_classification(Y, emb)

        return f1, acc, auc, nmi, mi_f1, ma_f1
    elif(dataset == "aminer"):
        Y = aminer_evaluation.aminer_evaluation()
        f1, acc, auc = aminer_link_prediction(Y, emb)
        nmi = aminer_cluster(Y, emb)
        mi_f1, ma_f1 = aminer_classification(Y, emb)

        return f1, acc, auc, nmi, mi_f1, ma_f1


    else:
        Y = acm_evaluation.Acm_evaluation()
        f1, acc, auc = acm_link_prediction(Y, emb)
        nmi = acm_cluster(Y, emb)
        mi_f1, ma_f1 = acm_classification(Y, emb)

        return f1, acc, auc, nmi, mi_f1, ma_f1

