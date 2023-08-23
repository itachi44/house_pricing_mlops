from sklearn.model_selection import RandomizedSearchCV

def optimize_model(X_train, y_train, model, param_dist, n_iter=100, cv=5, random_state=None, n_jobs=-1):
    """
    Recherche les meilleurs paramètres pour un modèle à l'aide de RandomizedSearchCV.

    Paramètres :
    ------------
    X_train : array-like
        Les caractéristiques d'entraînement.

    y_train : array-like
        Les valeurs cibles d'entraînement.

    best_model : Estimator object
        Le modèle de base pour lequel vous souhaitez trouver les meilleurs paramètres.

    param_dist : dict
        Les distributions des hyperparamètres à explorer.

    n_iter : int, optional (par défaut=100)
        Le nombre d'itérations de recherche aléatoire.

    cv : int, optional (par défaut=5)
        Le nombre de folds de validation croisée à utiliser.

    random_state : int ou RandomState, optional (par défaut=None)
        Contrôle la randomisation pour la reproductibilité.

    n_jobs : int, optional (par défaut=-1)
        Le nombre de tâches à exécuter en parallèle. -1 signifie utiliser tous les cœurs disponibles.

    Renvoie :
    ---------
    best_estimator : Estimator object
        Le meilleur modèle entraîné avec les meilleurs paramètres.

    best_params : dict
        Les meilleurs paramètres trouvés par la recherche aléatoire.
    """

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=n_iter, scoring="neg_mean_absolute_error", cv=cv,
                                       random_state=random_state, n_jobs=n_jobs, verbose=False)

    # Fit the RandomizedSearchCV object to your data
    random_search.fit(X_train, y_train)

    # Get the best parameters and best estimator from RandomizedSearchCV
    best_params = random_search.best_params_
    best_estimator = random_search.best_estimator_

    # Train the best estimator on the entire training set
    best_estimator.fit(X_train, y_train)

    return best_estimator, best_params
