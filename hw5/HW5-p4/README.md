# CSE_5243 HW5

In the folder, there are:

```
exact_similarity.py - compute the exact Jaccard similarity.

minhash_similarity.py - compute k-minhash signatures and estimated Jaccard similarity.

main.py - generate binary features and call similartity-computing functions mentioned above

eval.py - generate viualization of similarity matrices and calculate the mean-squared error

utils.py - contain the Indexer class

report.pdf - a detailed report, listing all underlying assumptions (if any), and explanations for performance obtained is expected.

README.md - A detailed README file that contains all the information about the folder
```

The way of running my program:

```
python main.py
python eval.py
```

After running "main.py", the calculated similarity matrices are saved to disk. 
By running "eval.py", we read and visualize the calculated similarity matrices and calculate the mean-squared error. 
You can see the rendered matrices and listed error. 
