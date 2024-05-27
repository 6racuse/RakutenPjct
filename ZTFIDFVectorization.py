def Vectorize_matrix(X_dt):
    """
    This function vectorizes the input data using TF-IDF Vectorizer.

    Args:
        X_dt (iterable): The input data to be vectorized. It should be an iterable of strings.

    Returns:
        X_tfidf_matrix (sparse matrix, [n_samples, n_features]): Transformed input data. Each row represents a document, and each column represents a feature.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf = TfidfVectorizer()
    X_tfidf_matrix = tfidf.fit_transform(X_dt)
    return X_tfidf_matrix