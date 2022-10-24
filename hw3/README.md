# CSE_5243 HW3

In the folder, there are:

```
*.py - source code 

report.pdf - a detailed report, listing all underlying assumptions (if any), and explanations for performance obtained is expected.

README.md - A detailed README file that contains all the information about the folder
```

The code support two feature vectors: the original feature vector from homework 2 (UNIGRAM) and the pared-down feature vector;
and three classifers: KNN, decision tree, and logistic regression. 

Here are some examples showing how to run the progarm. 

(1) The way of running KNN with the original feature vector:
```
python sensiment_classifier.py --model KNN \
                               --k K \
                               --feats UNIGRAM
```

(2) The way of running KNN with the pared-down feature vector:
```
python sensiment_classifier.py --model KNN \
                               --k K \
                               --feats IMPORTANT \
                               --appear 2
```

(3) The way of running decision tree with the original feature vector:
```
python sensiment_classifier.py --model DT \
                               --feats UNIGRAM
```

(4) The way of running logistic regression with the original feature vector:
```
python sensiment_classifier.py --model LR \
                               --feats UNIGRAM
```

The way of interpreting the output of my code:

In the program, I first output the time it would take for training.
Then I output the accuracy, precision, recall, and F1 for the training, validation and testing set. 
Finally, I report the evaluation time.
