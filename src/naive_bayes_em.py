import warnings
import numpy as np

from src.utils import softmax, stable_log_sum
from src.naive_bayes import NaiveBayes


class NaiveBayesEM(NaiveBayes):
    """
    A NaiveBayes classifier for binary data, that uses both unlabeled and
        labeled data in the Expectation-Maximization algorithm

    Note that the class definition above indicates that this class
        inherits from the NaiveBayes class. This means it has the same
        functions as the NaiveBayes class unless they are re-defined in this
        function. In particular you should be able to call `self.predict_proba`
        using your implementation from `src/naive_bayes.py`.
    """

    def __init__(self, max_iter=10, smoothing=1):
        """
        Args:
            max_iter: the maximum number of iterations in the EM algorithm,
                where each iteration contains both an E step and M step.
                You should check for convergence after each iterations,
                e.g. with `np.isclose(prev_likelihood, likelihood)`, but
                should terminate after `max_iter` iterations regardless of
                convergence.
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.max_iter = max_iter
        self.smoothing = smoothing

    def initialize_params(self, vocab_size, n_labels):
        """
        Initialize self.alpha such that
            `log p(y_i = k) = -log(n_labels)`
            for all k
        and initialize self.beta such that
            `log p(w_j | y_i = k) = -log(vocab_size)`
            for all j, k.

        """

        self.beta = np.zeros((vocab_size, n_labels))
        self.alpha = np.zeros((n_labels))

        self.alpha[:] = -np.log(n_labels)
        self.beta[:,:] = -np.log(vocab_size)

        print(f"self.alpha = {self.alpha}, self.beta = {self.beta}")

        # print(f"START - self.alpha = {self.alpha}, self.beta = {self.beta}")

        # raise NotImplementedError

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        You should store log probabilities to avoid underflow.
        This function *should* use unlabeled data within the EM algorithm.

        During the E-step, use the NaiveBayes superclass self.predict_proba to
            infer a distribution over the labels for the unlabeled examples.
            Note: you should *NOT* replace the true labels with your predicted
            labels. You can use a `np.where` statement to only update the
            labels where `np.isnan(y)` is True.

        During the M-step, update self.alpha and self.beta, similar to the
            `fit()` call from the NaiveBayes superclass. However, when counting
            words in an unlabeled example to compute p(x | y), instead of the
            binary label y you should use p(y | x).

        For help understanding the EM algorithm, refer to the lectures and
            the handout. In particular, Figure 2 shows the algorithm for
            semi-supervised Naive Bayes.

        self.alpha should contain the marginal probability of each class label.

        self.beta should contain the conditional probability of each word
            given the class label: p(x | y). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total
            words across all documents with label y=1, have a vocabulary size
            of V words, and see the word "jackpot" `k` times, then:
            `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing * V)`
            Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Note: if self.max_iter is 0, your function should call
            `self.initialize_params` and then break. In each
            iteration, you should complete both an E-step and
            an M-step.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        # # print(f"self.max_iter = {self.max_iter}")
        # if self.max_iter == 0:
        #     self.initialize_params(vocab_size, n_labels)
        #     return

        # else:
        #     # print(f"\n X = {X.toarray()}, \n y = {y}")
        #     self.initialize_params(vocab_size, n_labels)
        #     # print(f"\n n_docs = {n_docs}, vocab_size = {vocab_size}, START - self.alpha = {self.alpha}, self.beta = {self.beta}")
        #     for t in range(self.max_iter):

        #         # E-Step
        #         for i in range(n_labels):
        #             probs_label = self.predict_proba(X) # Get labels for unlabeled data
        #             # print(f"probs_label = {probs_label}")
        #             if np.isnan(y[i]) == True: 
        #                 pass

        #         # M-Step
        #         # Alpha
        #         for a in range(n_labels):
        #             sum = 0
        #             for n in range(n_labels):
        #                 if y[n] == a:
        #                     sum += 1
        #             self.alpha[a] = np.log(sum/n_labels)

        #         # Beta
        #         for jb in range(self.beta.shape[0]):
        #             for yb in range(self.beta.shape[1]):
        #                 num = 0
        #                 den = 0
        #                 for i in range(n_docs):
        #                     if y[i] == yb:
        #                         num += 1*X[i,jb]+self.smoothing

        #                     for j in range(vocab_size):
        #                         if y[i] == yb:
        #                             den += 1*X[i,j]+self.smoothing

        #                 self.beta[jb,yb] = np.log(num/den)

         # print(f"self.max_iter = {self.max_iter}")
        if self.max_iter == 0:
            self.initialize_params(vocab_size, n_labels)
            return

        else:
            # print(f"\n X = {X.toarray()}, \n y = {y}")
            self.initialize_params(vocab_size, n_labels)
            # print(f"\n n_docs = {n_docs}, vocab_size = {vocab_size}, START - self.alpha = {self.alpha}, self.beta = {self.beta}")
            self.y_new = np.copy(y)

            for t in range(self.max_iter):



                for z in range(len(self.alpha)):

                    # E-Step
                    for i in range(len(y)):
                        # print(f"probs_label = {probs_label}")
                        if np.isnan(y[i]) == True:
                            if z == 0:
                                self.y_new[i] = self.predict_proba(X[i])[0][0] # Get prob for index 0
                            else:
                                self.y_new[i] = self.predict_proba(X[i])[0][1] # Get prob for index 1
                    # print(f"\n \n \n new_y = {new_y}")

                    # M-Step
                    # Alpha
                    # for a in range(n_labels):
                    sum = 0
                    for n in range(len(y)): #range(n_labels):
                        if self.y_new[n] == z and np.isnan(y[n]) == False:
                            sum += 1
                        elif np.isnan(y[n]) == True:
                            sum += self.y_new[n]
                    self.alpha[z] = np.log(sum/n_docs) # np.log(sum/n_labels)
                    
                # Beta
                for jb in range(self.beta.shape[0]):
                    for yb in range(self.beta.shape[1]):
                        num = 0
                        den = 0
                        for i in range(n_docs):
                            if y[i] == yb:
                                num += 1*X[i,jb]+self.smoothing

                            for j in range(vocab_size):
                                if y[i] == yb:
                                    den += 1*X[i,j]+self.smoothing

                        self.beta[jb,yb] = np.log(num/den)



        # raise NotImplementedError

    def likelihood(self, X, y):
        r"""
        Using fit self.alpha and self.beta, compute the likelihood of the data.
            This function *should* use unlabeled data.
            This likelihood is defined in equation (14) of `naive_bayes.pdf`.

        For unlabeled data, we predict `p(y_i = y' | X_i)` using the
            previously-learned p(x|y, beta) and p(y | alpha).
            For labeled data, we define `p(y_i = y' | X_i)` as
            1 if `y_i = y'` and 0 otherwise; this is because for labeled data,
            the probability that the ith example has label y_i is 1.

        Following equation (14) in the `naive_bayes.pdf` writeup, the log
            likelihood of the data after t iterations can be written as:

            \sum_{i=1}^N \log \sum_{y'=1}^2 \exp(
                \log p(y_i = y' | X_i, \alpha, \beta) + \alpha_{y'}
                + \sum_{j=1}^V X_{i, j} \beta_{j, y'})

            You can visualize this formula in http://latex2png.com

            The tricky aspect of this likelihood is that we are simultaneously
            computing $p(y_i = y' | X_i, \alpha^t, \beta^t)$ to predict a
            distribution over our latent variables (the unobserved $y_i$) while
            at the same time computing the probability of seeing such $y_i$
            using $p(y_i =y' | \alpha^t)$.

            Note: In implementing this equation, it will help to use your
                implementation of `stable_log_sum` to avoid underflow. See the
                documentation of that function for more details.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the log likelihood of the data.
        """

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

        likelihood = 0


        for i in range(n_docs):
            sum1 =0
            for k in range(2):
                sum2 = 0
                for j in range (vocab_size):
                    if X[i,j] == 0:
                        sum2 += 0
                    elif np.isinf(self.beta[j,k]) == True:
                        sum2 += -np.inf 
                    else:
                        sum2 += X[i,j]*self.beta[j,k]
            
                if np.isnan(y[i]):
                    sum1 += np.exp(np.log(self.y_new[i]) + self.alpha[k] + sum2)
                elif y[i] == k:
                    sum1 += np.exp(np.log(1) + self.alpha [k] + sum2)
                else:
                    sum1 += np.exp(self.alpha[k] + sum2)
            likelihood += np.log(sum1)
        
        return likelihood

        # raise NotImplementedError