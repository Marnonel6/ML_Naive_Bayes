import warnings
import numpy as np

from tests.utils import build_small_dataset

train_data, train_labels, test_data, test_labels = build_small_dataset()


def test_tiny_dataset_a():
    from src.naive_bayes import NaiveBayes
    help_test_tiny_dataset(NaiveBayes)


def test_tiny_dataset_b():
    from src.naive_bayes_em import NaiveBayesEM
    help_test_tiny_dataset(NaiveBayesEM)


def help_test_tiny_dataset(model):
    from src.utils import softmax
    from scipy.sparse import csr_matrix

    X = csr_matrix(np.array([
        [1, 2, 1, 0, 0],
        [0, 1, 0, 1, 1]]))
    y = np.array([0, 1])
    nb = model(smoothing=0)
    nb.fit(X, y)

    # alpha should be of shape [n_labels]
    assert nb.alpha.shape == (2, )
    # beta should be of shape [n_vocab, n_labels]
    assert nb.beta.shape == (5, 2)

    msg = f"{softmax(nb.alpha)} != [0.5, 0.5]"
    assert np.allclose(softmax(nb.alpha), np.array([0.5, 0.5])), msg
    msg = f"{nb.alpha} != [-0.693, -0.693]"
    assert np.allclose(nb.alpha, np.log(0.5) * np.ones_like(nb.alpha)), msg

    beta_target = np.transpose(np.array([
        [0.25, 0.5, 0.25, 0, 0],
        [0, 1/3, 0, 1/3, 1/3]]))
    msg = f"{softmax(nb.beta, axis=0)} != {beta_target}"
    assert np.allclose(softmax(nb.beta, axis=0), beta_target), msg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        msg = f"{nb.beta} != {np.log(beta_target)}"
        assert np.allclose(nb.beta, np.log(beta_target)), msg

    # Log likelihood should match reference output
    assert np.isclose(nb.likelihood(X, y), -8.84101431048389)

    nb = model(smoothing=1)
    nb.fit(X, y)
    probs = np.array([[0.84891192, 0.15108808],
                      [0.20846906, 0.79153094]])

    # Predicted probabilities should match reference
    msg = f"{nb.predict_proba(X)} != {probs}"
    assert np.all(np.isclose(nb.predict_proba(X), probs)), msg


def test_smoothing():
    from src.naive_bayes import NaiveBayes
    from scipy.sparse import csr_matrix

    X = csr_matrix(np.array([
        [1, 2, 1, 0, 0],
        [0, 1, 0, 1, 1]]))
    train_y = np.array([0, 1])
    nb = NaiveBayes(smoothing=0)
    nb.fit(X, train_y)
    test_y = np.array([1, 0])

    # The log likelihood should be log(0) = -np.inf
    #   this happens because smoothing = 0, train_y != test_y,
    #   and some words only show up in one of the two documents.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        assert nb.likelihood(X, test_y) == -np.inf

    prev_prob = -np.inf
    for smoothing in [1, 2, 4, 1e100]:
        nb = NaiveBayes(smoothing=smoothing)
        nb.fit(X, train_y)
        prob = np.mean(nb.predict_proba(X)[(0, 1), (1, 0)])

        # The probability of seeing the opposite class should keep
        #     increasing as we increase the smoothing parameter
        msg = f"With smoothing={smoothing}, expect {prob} > {prev_prob}"
        assert prob > prev_prob, msg
        prev_prob = prob

    # When smoothing is near-infinite, probabilities should all be 0.5
    assert np.isclose(prob, 0.5)


def test_without_em():
    from src.naive_bayes import NaiveBayes

    # Train and evaluate NB without EM
    nb = NaiveBayes()
    nb.fit(train_data, train_labels)
    nb_likelihood = nb.likelihood(train_data, train_labels)

    is_labeled = np.isfinite(train_labels)
    nb_preds = nb.predict(train_data[is_labeled, :])
    train_accuracy = np.mean(nb_preds == train_labels[is_labeled])

    # NB should get 100% accuracy on the two labeled examples
    assert train_accuracy == 1.0

    nb_probs = nb.predict_proba(train_data)
    # Predict_proba should output a [n_documents, n_labels] array
    assert nb_probs.shape == (train_labels.shape[0], 2)
    # Probabilities should sum to 1
    assert np.all(np.isclose(np.sum(nb_probs, axis=1), np.ones_like(train_labels)))


def test_em_initialization():
    from src.utils import softmax
    from src.naive_bayes_em import NaiveBayesEM

    nbem = NaiveBayesEM(max_iter=0)
    nbem.initialize_params(train_data.shape[1], 2)
    assert np.isclose(np.sum(softmax(nbem.alpha, axis=1)), 1)
    assert np.allclose(np.sum(softmax(nbem.beta, axis=0), axis=0), 1)

    nbem.fit(train_data, train_labels)

    # If you do zero EM steps, your initialized probabilities should be uniform
    assert np.all(nbem.alpha[0] == nbem.alpha)
    assert np.all(nbem.beta[0, :] == nbem.beta)


def test_em_likelihood_always_increases():
    from src.naive_bayes_em import NaiveBayesEM
    prev_likelihood = -np.inf
    for max_iter in [1, 2, 3, 4, 5]:
        nb = NaiveBayesEM(max_iter=max_iter)
        nb.fit(train_data, train_labels)
        likelihood = nb.likelihood(train_data, train_labels)

        # EM should only ever increase the likelihood
        assert likelihood >= prev_likelihood
        prev_likelihood = likelihood


def test_comparison_naive_bayes():
    from src.naive_bayes import NaiveBayes
    from src.naive_bayes_em import NaiveBayesEM

    # Train and evaluate NB without EM
    nb1 = NaiveBayes()
    nb1.fit(train_data, train_labels)
    nb1_likelihood = nb1.likelihood(train_data, train_labels)
    nb1_preds = nb1.predict(test_data)
    nb1_accuracy = np.mean(nb1_preds == test_labels)

    # Train and evaluate NB with EM
    nb2 = NaiveBayesEM()
    nb2.fit(train_data, train_labels)
    nb2_likelihood = nb2.likelihood(train_data, train_labels)
    nb2_preds = nb2.predict(test_data)
    nb2_accuracy = np.mean(nb2_preds == test_labels)

    # NB using EM should outperform NB without it
    assert nb2_accuracy > nb1_accuracy

    # NB with EM should have a lower likelihood. Why? We'll see in the FRQ
    assert nb2_likelihood < nb1_likelihood
