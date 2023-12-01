
import argparse
import os
import pickle
import json
import numpy as np
from random import random
import re
import pandas as pd
import nltk

import stanza
stanza.download('en')
nlp = stanza.Pipeline('en')

#from cmn.review import Review

#* read pickle of review objs 
#* for fold i in splits, write corresponding reviews to test, valid, f{i}_train jsons ? so these jsons can be used by emc -- converts lady dataset into form readable by emc, so we can run emc from lady

#?dev.json equiv to valid.json ? yes

def load(reviews, splits):
    print('\n Loading reviews and preprocessing ...')
    print('#' * 50)
    try:
        print('\nLoading reviews file ...')
        with open(f'{reviews}', 'rb') as f:
            reviews = pickle.load(f)
        with open(f'{splits}', 'r') as f:
            splits = json.load(f)
    except (FileNotFoundError, EOFError) as e:
        print(e)
        print('\nLoading existing file failed!')
    print(f'(#reviews: {len(reviews)})')
    return reviews, splits


def get_aos_augmented(review):
    r = []
    if not review.aos: return r
    for i, aos in enumerate([review.aos]): r.append(
        [([review.sentences[i][j] for j in a], [review.sentences[i][j] for j in o], s) for (a, o, s) in aos])       #? what is this appending?
    return r

#ex review from train.json
"""
{"id": "0", 
 "sentence": "But the staff was so horrible to us .", 
 "triples": [
    {"uid": "0-0", 
    "sentiment": "negative", 
    "target_tags": "But\\O the\\O staff\\B was\\O so\\O horrible\\O to\\O us\\O .\\O", 
    "opinion_tags": "But\\O the\\O staff\\O was\\O so\\O horrible\\B to\\O us\\O .\\O"}], 
 "postag": ["CC", "DT", "NN", "VBD", "RB", "JJ", "IN", "PRP", "."], 
 "head": [6, 3, 6, 6, 6, 0, 8, 6, 6], 
 "deprel": ["cc", "det", "nsubj", "cop", "advmod", "root", "case", "obl", "punct"]}
"""

#? how do i generate postag, etc?       // see data.py, create instance and use functions to generate postag etc. or see nltk library
#? what exactly does aos() return? just [a,o,s] as expected?    see txt, returns similar list. [(a,o,s), (a,o,s)] --> multiple if review has multiple aspects etc.
#? difference between elements in triples?                      see above
#? "push and update emc baseline" -- what exactly do i need done? just wrapper or whole integration with lady?      --> just finish wrapper - ask drfani to make fork of emc in fanis lab so i can push to it, or he can just get wrapper from me.

def preprocess(org_reviews, is_test, lang, fname):
    reviews_list = []
    
    for r in org_reviews:
        review_info = r.to_dict()          #LADy Review function; gives id, text, sentences, aos, lang, orig
        r_dict = dict.fromkeys(["id", "sentence", "triples", "postag", "head", "deprel"]) 
        r_dict = {"id": review_info["id"],
                "sentence": review_info["text"],
                "triples": []
        }
        
        for j, aos in enumerate(review_info["aos"]): 
            triple = dict.fromkeys(["uid", "sentiment", "target_tags", "opinion_tags"])   
            triple["uid"] = r_dict["id"]+"-"+str(j)
            triple["sentiment"] = aos[2]

            t_tags = review_info["text"].split(" ")
            for i in range(len(t_tags)):
                if i == aos[0]:
                    t_tags[i] += "\\B"
                else:
                    t_tags[i] += "\\0"
            o_tags = review_info["text"].split(" ")
            for i in range(len(t_tags)):
                if i == aos[1]:
                    t_tags[i] += "\\B"
                else:
                    t_tags[i] += "\\0"

            triple["target_tags"] = t_tags
            triple["target_tags"] = o_tags

            r_dict["triples"].append(triple)

        #data.py/Instance gens deprel, postag, etc in init (?)
        #r_instance = Instance()

        #..try nltk instead
        postag_pairs = nltk.pos_tag(review_info["sentences"])
        postags = [postag_pair[1] for postag_pair in postag_pairs]
        r_dict["postag"] = postags

        #using stanza -- idk ? https://towardsdatascience.com/natural-language-processing-dependency-parsing-cf094bbbe3f7
        doc = nlp(review_info["sentences"])
        r_dict["deprel"] = doc["deprel"]
        r_dict["head"] = doc["head"]

        

        reviews_list.append(r_dict)
        
    reviews_json = json.dumps(reviews_list)             #json of list of dicts [one per review] to be written to json file
    return reviews_json

def writeToJSON(dct, fname):
    with open(f'../output/{fname}.json', 'w') as fp:
        json.dump(dct, fp)



#! modify this to look like emc's jsons ..^
"""
def preprocess(org_reviews, is_test, lang):
    reviews_list = []
    label_list = []
    for r in org_reviews:
        if not len(r.aos[0]):
            continue
        else:
            aos_list = []
            if r.augs and not is_test:                  #? r.augs is a dictionary? ? 
                if lang == 'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn':
                    for key, value in r.augs.items():
                        aos_list = []
                        text = ' '.join(r.augs[key][1].sentences[0]).strip() + '####'
                        for aos_instance in r.augs[key][1].aos:
                            aos_list.extend(aos_instance[0])
                        for idx, word in enumerate(r.augs[key][1].sentences[0]):
                            if idx in aos_list:
                                tag = word + '=T-POS' + ' '
                                text += tag
                            else:
                                tag = word + '=O' + ' '
                                text += tag
                        if len(text.rstrip()) > 511:
                            continue
                        reviews_list.append(text.rstrip())
                else:
                    for aos_instance in r.augs[lang][1].aos:
                        aos_list.extend(aos_instance[0])
                    text = ' '.join(r.augs[lang][1].sentences[0]).strip() + '####'
                    for idx, word in enumerate(r.augs[lang][1].sentences[0]):
                        if idx in aos_list:
                            tag = word + '=T-POS' + ' '
                            text += tag
                        else:
                            tag = word + '=O' + ' '
                            text += tag
                    if len(text.rstrip()) > 511:
                        continue
                    reviews_list.append(text.rstrip())

            aos_list = []
            for aos_instance in r.aos[0]:
                aos_list.extend(aos_instance[0])
            # text = ' '.join(r.sentences[0]).replace('*****', '').replace('   ', '  ').replace('  ', ' ').replace('  ', ' ') + '####'
            # text = re.sub(r'\s{2,}', ' ', ' '.join(r.sentences[0]).replace('#####', '').strip()) + '####'
            text = re.sub(r'\s{2,}', ' ', ' '.join(r.sentences[0]).strip()) + '####'
            for idx, word in enumerate(r.sentences[0]):
                # if is_test and word == "#####":
                #     continue
                if idx in aos_list:
                    tag = word + '=T-POS' + ' '
                    text += tag
                else:
                    tag = word + '=O' + ' '
                    text += tag
            if len(text.rstrip()) > 511:
                continue
            reviews_list.append(text.rstrip())
            aos_list_per_review = []
            for idx, word in enumerate(r.sentences[0]):
                if idx in aos_list:
                    aos_list_per_review.append(word)
            label_list.append(aos_list_per_review)
    return reviews_list, label_list
"""


# python main.py -ds_name [YOUR_DATASET_NAME] -sgd_lr [YOUR_LEARNING_RATE_FOR_SGD] -win [YOUR_WINDOW_SIZE] -optimizer [YOUR_OPTIMIZER] -rnn_type [LSTM|GRU] -attention_type [bilinear|concat]
def main(args):
    output_path = f'{args.output}/{args.dname}'
    print(output_path)
    # if not os.path.isdir(output_path): os.makedirs(output_path)
    org_reviews, splits = load(args.reviews, args.splits)

    test = np.array(org_reviews)[splits['test']].tolist()
    test = preprocess(test, True, args.lang)
    writeToJSON(train, "train")
    for h in range(0, 101, 10):
        hp = h / 100

        for f in range(5):
            train = preprocess(np.array(org_reviews)[splits['folds'][str(f)]['train']].tolist(), False, args.lang)       
            writeToJSON(train, f"train{f}")
            dev = preprocess(np.array(org_reviews)[splits['folds'][str(f)]['valid']].tolist(), False, args.lang, f"dev{f}")
            writeToJSON(dev, f"dev{f}")
            path = f'{output_path}-fold-{f}-latency-{h}'
            if not os.path.isdir(path): os.makedirs(path)

            with open(f'{path}/dev.txt', 'w', encoding='utf-8') as file:
                for d in dev:
                    file.write(d + '\n')

            with open(f'{path}/train.txt', 'w', encoding='utf-8') as file:
                for d in train:
                    file.write(d + '\n')

            test_hidden = []
            for t in range(len(test)):
                if random() < hp:
                    test_hidden.append(test[t].hide_aspects(mask="z", mask_size=5))
                else:
                    test_hidden.append(test[t])
            preprocessed_test, _ = preprocess(test_hidden, True, args.lang)

            with open(f'{path}/test.txt', 'w', encoding='utf-8') as file:
                for d in preprocessed_test:
                    file.write(d + '\n')

            pd.to_pickle(labels, f'{path}/test-labels.pk')
            # with open(f'{path}/test-labels.txt', 'w', encoding='utf-8') as file:
            #     for label in labels:
            #         for l in label:
            #             file.write(l)
            #         file.write('\n')

    # for f in range(5):
    #     train = preprocess(np.array(org_reviews)[splits['folds'][str(f)]['train']].tolist(), False, args.lang)
    #     dev = preprocess(np.array(org_reviews)[splits['folds'][str(f)]['valid']].tolist(), False, args.lang)
    #     path = f'{output_path}-fold-{f}'
    #     if not os.path.isdir(path): os.makedirs(path)
    #
    #     with open(f'{path}/dev.txt', 'w') as file:
    #         for d in dev:
    #             file.write(d + '\n')
    #
    #     with open(f'{path}/train.txt', 'w') as file:
    #         for d in train:
    #             file.write(d + '\n')

    # repeat




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EMC Wrapper')
    parser.add_argument('--dname', dest='dname', type=str, default='16semeval_rest')
    parser.add_argument('--reviews', dest='reviews', type=str,
                        default='reviews.pkl',                                             
                        help='raw dataset file path')
    parser.add_argument('--splits', dest='splits', type=str,
                        default='splits.json',
                        help='raw dataset file path')
    parser.add_argument('--output', dest='output', type=str, default='data', help='output path')
    parser.add_argument('--lang', dest='lang', type=str, default='eng', help='language')

    args = parser.parse_args()

    for dataset in ['SemEval14L']:  # 'SemEval14L', 'SemEval14R', '2015SB12', '2016SB5'     #? should this be args.dname ?
        args.splits = f'data/{dataset}/splits.json'
        for lang in ['eng', 'spa_Latn', 'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn']:
            if lang == 'eng':
                args.lang = lang
                args.dname = f'{dataset}'
                args.reviews = f'data/{dataset}/reviews.pkl'
            else:
                args.lang = lang
                args.dname = f'{dataset}-{lang}'
                args.reviews = f'data/{dataset}/reviews.{lang}.pkl'
            print(args)
            main(args)
