def Vectorize_matrix(X_dt):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf = TfidfVectorizer()
    X_tfidf_matrix = tfidf.fit_transform(X_dt)
    return X_tfidf_matrix