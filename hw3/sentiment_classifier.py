# sentiment_classifier.py

import time
from models import *
from sentiment_data import *

if __name__ == '__main__':
    np.random.seed(0)

    train_items, dev_items, test_items = read_sentiment_examples()

    # Train and evaluate
    start_time = time.time()
    model = train_model(train_items)
