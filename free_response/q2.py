import numpy as np

from free_response.data import build_dataset
from src.naive_bayes_em import NaiveBayesEM


def main():
    """
    Helper code for the Free Response Q2b
    You will have to write your own code for Q2a
    """
    # Load the dataset; fit the NB+EM model
    data, labels, speeches, vocab = build_dataset(
        "data", num_docs=100, max_words=2000, vocab_size=1000)
    isfinite = np.isfinite(labels)
    nb = NaiveBayesEM(max_iter=10)
    nb.fit(data, labels)

    # Use predict_proba to see output probabilities
    probs = nb.predict_proba(data)[isfinite]
    preds = nb.predict(data)
    correct = preds[isfinite] == labels[isfinite]

    # The model's "confidence" in its predicted output when right 
    right_label = labels[isfinite][correct].astype(int)
    prob_when_correct = probs[correct, right_label]

    # The model's "confidence" in its predicted output when wrong 
    incorrect = np.logical_not(correct)
    wrong_label = 1 - labels[isfinite][incorrect].astype(int)
    prob_when_incorrect = probs[incorrect, wrong_label]

    # Use these number to answer FRQ 2b
    print("When NBEM is correct:")
    print(prob_when_correct.tolist())
    print("When NBEM is incorrect:")
    print(prob_when_incorrect.tolist())


if __name__ == "__main__":
    main()
