import senteval
import sys
import io
import os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = ''
PATH_TO_DATA = ''

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def main(modelName):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str,
                        choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'],
                        default='cls',
                        help="Which pooler to use")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+',
                        default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                                 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                                 'SICKRelatedness', 'STSBenchmark'],
                        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")

    args = parser.parse_args()

    # Load transformers' model checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(device, modelName)

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        return sentence2feature(model, sentences, device, max_length)

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


def get_clip_model(device):
    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    from clip import clip
    clip_model, preprocess = clip.load("ViT-L/14@336px", device=device, jit=False)
    clip_model.eval()
    return clip_model


def sentence2feature_clip(model, sentences, device, max_length):
    with torch.no_grad():
        from clip import clip
        if max_length is None:
            tokens = clip.tokenize(sentences, truncate=True)
        else:
            tokens = clip.tokenize(sentences, context_length=max_length, truncate=True)
        tokens = tokens.to(device)
        return model.encode_text(tokens).cpu()


def sentence2feature_mpnet(model, sentences, device, max_length):
    caption_features = model.encode(sentences, device=device, show_progress_bar=False, convert_to_tensor=True).to(device)
    return caption_features.cpu()


def get_mpnet_model(device):
    from sentence_transformers import SentenceTransformer
    txt_enc_assisant = SentenceTransformer('all-mpnet-base-v2').to(device=device)
    return txt_enc_assisant

def get_model(device, modelName):
    # return get_clip_model(device)
    # return get_mpnet_model(device)
    from unire.model import unire
    checkPointPath = modelName
    print("Loading model: ", checkPointPath)
    checkpoint = torch.load(checkPointPath, map_location='cpu')
    state_dict = checkpoint['model']
    args = argparse.Namespace()
    args.gpu = torch.device(device)
    model = unire(args, checkpoint['config'])
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def sentence2feature(model, sentences, device, max_length):
    # return sentence2feature_clip(model, sentences, device, max_length)
    # return sentence2feature_mpnet(model, sentences, device, max_length)
    with torch.no_grad():
        from clip import clip
        if max_length is None:
            tokens = clip.tokenize(sentences, truncate=True)
        else:
            tokens = clip.tokenize(sentences, context_length=max_length, truncate=True)
        tokens = tokens.to(device)
        # return model.encode_text(tokens, cross_modal=True).cpu()
        return model.encode_text(tokens, cross_modal=False).cpu()


if __name__ == "__main__":
    processList = []
    fileList = list(os.walk("output/vitb32/coco"))
    for val in fileList:
        if "checkpoint_best.pth" in val[2]:
            processList.append(os.path.join(val[0], "checkpoint_best.pth"))
    for val in processList:
        main(val)
