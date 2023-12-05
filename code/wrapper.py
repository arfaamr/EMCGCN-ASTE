#wrapper to convert LADy's review pkls to JSON files for EMCGCN

import argparse
import os
import pickle
import json
import numpy as np
from random import random
import re
import pandas as pd

import nltk
#nltk.download('averaged_perceptron_tagger')            #?how to change download location
import stanza
#stanza.download('en')                                   #? should download every needed language?
nlp = stanza.Pipeline('en')

from cmn.review import Review

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
        [([review.sentences[i][j] for j in a], [review.sentences[i][j] for j in o], s) for (a, o, s) in aos]) 
    return r


def preprocess(org_reviews, language):
    reviews_list = []
    
    for i,r in enumerate(org_reviews):
        review_info = r.to_dict()[0]          #LADy Review function; gives id, text, sentences, aos, lang, orig
        r_dict = dict.fromkeys(["id", "sentence", "triples", "postag", "head", "deprel"]) 
        r_dict = {"id": str(i),
                "sentence": review_info["text"],
                "triples": []
        }

        for j, aos in enumerate(review_info["aos"][0]): 
            triple = dict.fromkeys(["uid", "sentiment", "target_tags", "opinion_tags"])   
            triple["uid"] = r_dict["id"]+"-"+str(j)
            triple["sentiment"] = "negative" if aos[2]=="-1" else "neutral" if aos[2]=="0" else "positive"

            t_tags = review_info["text"].split(" ")
            for k in range(len(t_tags)):
                if t_tags[k] == aos[0][0]:              #based on comparison with some reviews in emc jsons, only aos[0] is used. aos[0][0] is aspect
                    t_tags[k] += "\\B"
                else:
                    t_tags[k] += "\\0"
            t_tags = " ".join(t_tags)
            triple["target_tags"] = t_tags

            o_tags = review_info["text"].split(" ")
            for k in range(len(o_tags)):
                if o_tags[k] == aos[1]:                 #aos[1] is opinion. but LADy's aos function is not giving opinions correctly? all opinions are blank
                    o_tags[k] += "\\B"
                else:
                    o_tags[k] += "\\0"
            o_tags = " ".join(o_tags)
            triple["opinion_tags"] = o_tags

            r_dict["triples"].append(triple)

        postag_pairs = nltk.pos_tag(review_info["sentences"][0])        #[0] because sentences is a list of a list of words      #! can add parameter lang = language, but nltk only supports english, russian
        postags = [postag_pair[1] for postag_pair in postag_pairs]
        r_dict["postag"] = postags

        #using stanza -- idk ? https://towardsdatascience.com/natural-language-processing-dependency-parsing-cf094bbbe3f7
        doc = nlp(review_info["text"])
        sent_dict = doc.sentences[0].to_dict()

        head = []
        deprel = []
        for word in sent_dict:
            head.append(word["head"])
            deprel.append(word['deprel'])
        r_dict["head"] = head
        r_dict["deprel"] = deprel

        reviews_list.append(r_dict)
        
    return reviews_list

def writeToJSON(lst, fname):
    with open(f'../output/{fname}.json', 'w') as fp:
        json.dump(lst, fp)


# python main.py -ds_name [YOUR_DATASET_NAME] -sgd_lr [YOUR_LEARNING_RATE_FOR_SGD] -win [YOUR_WINDOW_SIZE] -optimizer [YOUR_OPTIMIZER] -rnn_type [LSTM|GRU] -attention_type [bilinear|concat]
def main(args):
    output_path = f'{args.output}/{args.dname}'
    print(output_path)
    # if not os.path.isdir(output_path): os.makedirs(output_path)
    org_reviews, splits = load(args.reviews, args.splits)

    test = np.array(org_reviews)[splits['test']].tolist()
    test = preprocess(test, args.lang)
    writeToJSON(test, "test")
    for h in range(0, 101, 10):
        hp = h / 100

        for f in range(5):
            train = preprocess(np.array(org_reviews)[splits['folds'][str(f)]['train']].tolist(), args.lang)       
            writeToJSON(train, f"train{f}")
            dev = preprocess(np.array(org_reviews)[splits['folds'][str(f)]['valid']].tolist(), args.lang)
            writeToJSON(dev, f"dev{f}")

            #? what is this for
            #test_hidden = []
            #for t in range(len(test)):
            #    if random() < hp:
            #        test_hidden.append(test[t].hide_aspects(mask="z", mask_size=5))
            #    else:
            #        test_hidden.append(test[t])
            #preprocessed_test, _ = preprocess(test_hidden, True, args.lang)
            #
            #with open(f'{path}/test.txt', 'w', encoding='utf-8') as file:
            #    for d in preprocessed_test:
            #        file.write(d + '\n')

            #pd.to_pickle(labels, f'{path}/test-labels.pk')
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

    for dataset in ['SemEval14L']:  # 'SemEval14L', 'SemEval14R', '2015SB12', '2016SB5'
        #args.splits = f'data/{dataset}/splits.json'
        args.splits = "splits.json"     #* temp hardcoding args.splits, args.reviews to same path for testing
        for lang in ['eng', 'spa_Latn', 'pes_Arab.zho_Hans.deu_Latn.arb_Arab.fra_Latn.spa_Latn']:
            if lang == 'eng':
                args.lang = lang
                args.dname = f'{dataset}'
                args.reviews = 'reviews.pkl'
            else:
                args.lang = lang
                args.dname = f'{dataset}-{lang}'
                #args.reviews = f'data/{dataset}/reviews.{lang}.pkl'
                args.reviews = "reviews.pkl"
            print(args)
            main(args)
