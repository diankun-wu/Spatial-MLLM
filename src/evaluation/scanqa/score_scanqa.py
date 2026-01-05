import argparse
import json
import os
import re
import string

import matplotlib.pyplot as plt


def main(args):
    with open(args.input_file) as f:
        data = json.load(f)

    with open("datasets/evaluation/scanqa/scanqa_val_llava_style.json") as f:
        raw_data = json.load(f)
        idx2labels = {}
        for item in raw_data:
            idx2labels[item['id']] = item['metadata']['answers']


    from caption_eval.bleu.bleu import Bleu
    from caption_eval.cider.cider import Cider
    from caption_eval.meteor.meteor import Meteor
    from caption_eval.rouge.rouge import Rouge

    cider = Cider()
    bleu = Bleu()
    meteor = Meteor()
    rouge = Rouge()

    n_correct = 0
    res, gts = {}, {}
    for item in data:
        item["sample_id"] = "_".join(item["sample_id"].split("_")[:-1] + ['0'])
        res[item['sample_id']] = [item['pred_response'].rstrip(".")]
        gts[item['sample_id']] = idx2labels[item['sample_id']]

        if item['pred_response'] in idx2labels[item['sample_id']]:
            n_correct += 1

    cider_score = cider.compute_score(gts, res)
    bleu_score = bleu.compute_score(gts, res)
    meteor_score = meteor.compute_score(gts, res)
    rouge_score = rouge.compute_score(gts, res)

    print(f"count: {len(gts)}")
    print(f"CIDER: {cider_score[0]*100}")
    print(f"BLEU: {bleu_score[0][0]*100}, {bleu_score[0][1]*100}, {bleu_score[0][2]*100}, {bleu_score[0][3]*100}")
    print(f"METEOR: {meteor_score[0]*100}")
    print(f"Rouge: {rouge_score[0]*100}")
    print(f"EM: {n_correct / len(data)}")




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True, help='Path to the model predictions json file.')
    args = parser.parse_args()

    main(args)