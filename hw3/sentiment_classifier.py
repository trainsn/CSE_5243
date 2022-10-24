# sentiment_classifier.py

import argparse
import time
from models import *
from sentiment_data import *

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='KNN', help='model to run (KNN or LR)')
    parser.add_argument('--k', type=int, default=5, help='parameter k for KNN')
    parser.add_argument('--feats', type=str, default='UNIGRAM', help='feats to use (UNIGRAM, IMPORTANT)')
    parser.add_argument('--appear', type=int, default=2, help='the number a word should appear at least')
    args = parser.parse_args()
    return args

def evaluate(classifier, exs):
    """
    Evaluates a given classifier on the given examples
    :param classifier: classifier to evaluate
    :param exs: the list of SentimentExamples to evaluate on
    :return: None (but prints output)
    """
    print_evaluation([ex.label for ex in exs], [classifier.predict(ex.words) for ex in exs])


def print_evaluation(golds, predictions):
    """
    Prints evaluation statistics comparing golds and predictions, each of which is a sequence of 0/1 labels.
    Prints accuracy as well as precision/recall/F1 of the positive class, which can sometimes be informative if either
    the golds or predictions are highly biased.

    :param golds: gold labels
    :param predictions: pred labels
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision (fraction of predicted positives that are correct): %i / %i = %f" % (num_pos_correct, num_pred, prec)
          + "; Recall (fraction of true positives predicted correctly): %i / %i = %f" % (num_pos_correct, num_gold, rec)
          + "; F1 (harmonic mean of precision and recall): %f" % f1)

if __name__ == '__main__':
    args = _parse_args()
    print(args)

    np.random.seed(0)

    train_items, dev_items, test_items = read_sentiment_examples()

    # Train and evaluate
    start_time = time.time()
    model = train_model(args, train_items)
    print("Time for training: %.2f seconds" % (time.time() - start_time))

    start_time = time.time()
    print("=====Train Accuracy=====")
    evaluate(model, train_items)
    print("=====Dev Accuracy=====")
    evaluate(model, dev_items)
    print("=====Test Accuracy=====")
    evaluate(model, test_items)
    print("Time for evaluation: %.2f seconds" % (time.time() - start_time))


