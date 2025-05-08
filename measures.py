import numpy as np
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import networkx as nx
from sklearn.metrics import mean_squared_error
import itertools

def f1(X, y):
    """
    Calcula la medida F1 para clasificación multiclase (One-vs-One).
    Cada F1 indica el nivel de solapamiento por característica.
    
    Parámetros:
    X: np.array de forma [n_samples, n_features]
    y: np.array de etiquetas [n_samples]

    Retorna:
    f1_val: np.array de forma [n_features], con un F1 por característica
    """
    classes = np.unique(y)
    f1s = []

    # One-vs-One entre pares de clases
    for c1, c2 in itertools.combinations(classes, 2):
        sample_c1 = X[y == c1]
        sample_c2 = X[y == c2]

        mu1 = np.mean(sample_c1, axis=0)
        mu2 = np.mean(sample_c2, axis=0)

        var1 = np.var(sample_c1, axis=0)
        var2 = np.var(sample_c2, axis=0)

        fisher = (mu1 - mu2) ** 2 / (var1 + var2 + 1e-10)
        f1 = 1 / (1 + fisher)
        f1s.append(f1)

    # Promediar sobre todas las comparaciones entre pares de clases
    f1_val = np.mean(f1s, axis=0)
    return f1_val

def f1v(X, y):
    """
    Calcula F1v para clasificación multiclase utilizando el método One-vs-One.
    Basado en la fórmula de Malina [92] y Lorena et al. [88].

    Parámetros:
    X: np.array [n_samples, n_features]
    y: np.array [n_samples]

    Retorna:
    F1v: valor escalar, donde valores más altos indican más solapamiento
    """
    classes = np.unique(y)
    f1vs = []

    # Comparación One-vs-One entre todas las combinaciones de clases
    for c1, c2 in itertools.combinations(classes, 2):
        # Extraer los datos para las dos clases actuales
        X1 = X[y == c1]
        X2 = X[y == c2]
        
        # Calcular las medias por clase
        mu1 = np.mean(X1, axis=0)
        mu2 = np.mean(X2, axis=0)

        # Calcular las matrices de covarianza intra-clase
        S1 = np.cov(X1, rowvar=False)
        S2 = np.cov(X2, rowvar=False)
        Sw = S1 + S2  # Matriz de covarianza combinada

        # Entrenamos el LDA para obtener el vector de proyección óptimo
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X, y)
        w = lda.coef_[0]

        # Cálculo del criterio de Fisher direccional (dF)
        numerator = (w @ (mu1 - mu2)) ** 2
        denominator = w @ Sw @ w
        dF = numerator / (denominator + 1e-10)  # Añadir pequeño valor para evitar división por cero

        # Cálculo de F1v
        F1v = 1 / (1 + dF)
        f1vs.append(F1v)

    # Promediar los resultados de todas las comparaciones One-vs-One
    F1v_final = np.mean(f1vs)
    return F1v_final

# Función para calcular F2 (Volumen de la región superpuesta)
def f2(X, y):
    """
    Calcula F2: Volumen de la región superpuesta basado en distribuciones de características.
    """
    classes = np.unique(y)
    overlap_product = 1
    
    for feature in range(X.shape[1]):
        minmax = min(max(X[y == classes[0], feature]), max(X[y == classes[1], feature]))
        maxmin = max(min(X[y == classes[0], feature]), min(X[y == classes[1], feature]))
        range_all = max(X[:, feature]) - min(X[:, feature])
        
        overlap_ratio = max(0, minmax - maxmin) / range_all
        overlap_product *= overlap_ratio  # Producto de los solapamientos en cada característica
    
    return overlap_product

# Función para calcular F3 (Máxima eficiencia de funciones individuales)
def f3(X, y):
    """
    Calcula F3: Máxima eficiencia de características individuales para separar clases.
    """
    classes = np.unique(y)
    overlap_ratios = []
    
    for feature in range(X.shape[1]):
        minmax = min(max(X[y == classes[0], feature]), max(X[y == classes[1], feature]))
        maxmin = max(min(X[y == classes[0], feature]), min(X[y == classes[1], feature]))
        
        overlap_ratio = np.sum((X[:, feature] > maxmin) & (X[:, feature] < minmax)) / len(X)
        overlap_ratios.append(overlap_ratio)
    
    return min(overlap_ratios)  # Devuelve la mínima eficiencia (máximo solapamiento)

# Función para calcular F4 (Eficiencia de funciones colectivas)
def f4(X, y):
    """
    Calcula F4: Proporción de ejemplos no separados por ninguna característica.
    """
    classes = np.unique(y)
    remaining = np.ones(len(X), dtype=bool)  # Todos los ejemplos están inicialmente en el conjunto
    
    for feature in range(X.shape[1]):
        minmax = min(max(X[y == classes[0], feature]), max(X[y == classes[1], feature]))
        maxmin = max(min(X[y == classes[0], feature]), min(X[y == classes[1], feature]))
        
        # Remover ejemplos que ya están claramente separados en este feature
        remaining &= (X[:, feature] > maxmin) & (X[:, feature] < minmax)
    
    return np.sum(remaining) / len(X)  # Proporción de ejemplos aún solapados

# Función para calcular N1 (Fracción de puntos fronterizos)
def n1(X, y):
    """
    Calcula N1: Proporción de ejemplos conectados a la clase opuesta en un Árbol de Expansión Mínima (MST).
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    dist_matrix = distance.cdist(X, X)
    mst = minimum_spanning_tree(dist_matrix).toarray()
    border_points = set()
    for i in range(len(X)):
        for j in range(len(X)):
            if mst[i, j] > 0 and y[i] != y[j]:
                border_points.add(i)
                border_points.add(j)
    return len(border_points) / len(X)

# Función para calcular N2 (Razón de distancia intra/extra clase)
def n2(X, y):
    """
    Calcula N2: Relación entre la distancia intra-clase y la distancia inter-clase de los vecinos más cercanos.
    """
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    distances, indices = nn.kneighbors(X)
    intra_class_dists = []
    extra_class_dists = []
    for i in range(len(X)):
        for j in indices[i]:
            if i != j:
                if y[i] == y[j]:
                    intra_class_dists.append(distances[i][1])
                else:
                    extra_class_dists.append(distances[i][1])
    r = np.sum(intra_class_dists) / np.sum(extra_class_dists)
    return r / (1 + r)

# Función para calcular N3 (Tasa de error del clasificador 1-NN)
def n3(X, y):
    """
    Calcula N3: Tasa de error usando un clasificador 1NN con validación cruzada Leave-One-Out.
    """
    loo = LeaveOneOut()
    errors = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nn = NearestNeighbors(n_neighbors=1).fit(X_train)
        _, indices = nn.kneighbors(X_test)
        if y_train[indices[0][0]] != y_test[0]:
            errors += 1
    return errors / len(X)

# Función para calcular N4 (No linealidad del clasificador de vecinos más cercanos)
def n4(X, y):
    """
    Calcula N4: Mide el error del clasificador 1NN sobre ejemplos sintéticos interpolados.
    """
    synthetic_points = (X[:-1] + X[1:]) / 2
    nn = NearestNeighbors(n_neighbors=1).fit(X, y)
    _, indices = nn.kneighbors(synthetic_points)
    errors = sum(y[indices[:, 0]] != y[:-1])
    return errors / len(synthetic_points)

# Función para calcular T1 (Fracción de hiperesferas que cubren los datos)
def t1(X, y):
    """
    Calcula T1: Proporción de hiperesferas que cubren los datos, eliminando las redundantes.
    """
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    distances, indices = nn.kneighbors(X)
    hyperspheres = set()
    for i in range(len(X)):
        if y[i] != y[indices[i][1]]:
            hyperspheres.add(i)
    return len(hyperspheres) / len(X)

# Función para calcular LSC (Cardinalidad promedio del conjunto local)
def lsc(X, y):
    """
    Calcula LSC: Cardinalidad promedio del conjunto local.
    """
    nn = NearestNeighbors(n_neighbors=len(X)).fit(X)
    distances, indices = nn.kneighbors(X)
    lsc_values = []
    for i in range(len(X)):
        local_set = [j for j in indices[i] if y[j] == y[i] and distances[i][j] < distances[i][-1]]
        lsc_values.append(len(local_set))
    return np.mean(lsc_values)

def non_linearity_linear_classifier(X, y, n_samples=1000, random_state=42):
    """
    Estima la no-linealidad de un clasificador lineal (Logistic Regression)
    generando ejemplos sintéticos por interpolación lineal de pares de la misma clase
    y midiendo la tasa de error en dichos ejemplos.
    """
    rng = np.random.RandomState(random_state)
    synthetic_X = []
    synthetic_y = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        for _ in range(n_samples):
            i, j = rng.choice(idx, 2, replace=False)
            alpha = rng.rand()
            synthetic_X.append(alpha * X[i] + (1 - alpha) * X[j])
            synthetic_y.append(cls)
    synthetic_X = np.vstack(synthetic_X)
    synthetic_y = np.array(synthetic_y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    y_pred = clf.predict(synthetic_X)
    return np.mean(y_pred != synthetic_y)

def non_linearity_knn_classifier(X, y, n_samples=1000, random_state=42):
    """
    Estima la no-linealidad de un clasificador 1-KNN generando ejemplos sintéticos
    por interpolación lineal de pares de la misma clase y midiendo la tasa de error.
    """
    rng = np.random.RandomState(random_state)
    synthetic_X = []
    synthetic_y = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        for _ in range(n_samples):
            i, j = rng.choice(idx, 2, replace=False)
            alpha = rng.rand()
            synthetic_X.append(alpha * X[i] + (1 - alpha) * X[j])
            synthetic_y.append(cls)
    synthetic_X = np.vstack(synthetic_X)
    synthetic_y = np.array(synthetic_y)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    y_pred = knn.predict(synthetic_X)
    return np.mean(y_pred != synthetic_y)

def avg_samples_per_dimension(X):
    """
    Calcula el promedio de ejemplos por dimensión:
      número de muestras / número de características.
    """
    n_samples, n_features = X.shape
    return n_samples / n_features

def calc_network_measures(X, k=5):
    """
    Construye un grafo k-NN (aristas no dirigidas) y calcula:
      - Densidad de la red
      - Coeficiente de clustering medio
      - Puntajes de hubs (hub centrality)
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in indices[i][1:]:
            G.add_edge(i, j)
    
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    hubs, _ = nx.hits(G)  # hub and authority scores
    return {
        'density': density,
        'average_clustering': avg_clustering,
        'hub_scores': hubs
    }

def knn_regressor_error(X, y, k=5):
    """
    Estima el MSE de un regresor k-NN mediante validación Leave-One-Out.
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    predictions = []
    for i in range(n):
        Xi = np.delete(X, i, axis=0)
        yi = np.delete(y, i)
        knn = KNeighborsClassifier(n_neighbors=k)  # regression via classification interface
        knn.fit(Xi, yi)
        predictions.append(knn.predict(X[i].reshape(1, -1))[0])
    return mean_squared_error(y, predictions)

def input_distribution_measure(X, k=5):
    """
    Calcula el promedio de distancia a los k-vecinos más cercanos para cada punto.
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # ignoramos la distancia cero al mismo ejemplo:
    return np.mean(distances[:, 1:])

def output_distribution_measure(X, y, k=5):
    """
    Promedia la diferencia absoluta de las salidas entre cada punto y sus k-vecinos más cercanos.
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, indices = nbrs.kneighbors(X)
    diffs = []
    for i in range(n):
        neighbors = indices[i][1:]
        diffs.extend(np.abs(y[i] - y[j]) for j in neighbors)
    return np.mean(diffs)
