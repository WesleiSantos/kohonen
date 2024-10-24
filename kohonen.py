import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter



# Função para ler os dados
def read_data(filename):
    class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}  # Mapeia as classes para binário
    
    X = []
    d = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('@') or not line.strip():  # Ignora as linhas de metadados e vazias
                continue
            data = line.strip().split(',')
            X.append([float(x) for x in data[:-1]])  # Características
            d.append(class_mapping[data[-1].strip()])  # Classe convertida para valor binário
    return np.array(X), np.array(d)

# Função para calcular a distância euclidiana
def euclidean_distance(x, w):
    return np.sqrt(np.sum((x - w) ** 2, axis=1))

def learning_rate_decay(t, eta_0=0.1, tau_2=1000):
    """Calcula a taxa de aprendizado usando a fórmula η(t) = η0 * exp(-t / τ2)"""
    return eta_0 * np.exp(-t / tau_2)

# Função gaussiana para vizinhança (com tempo)
def gaussian_neighbor(dist_to_winner, sigma):
    """Calcula a influência gaussiana com base na distância do neurônio vencedor."""
    return np.exp(-(dist_to_winner**2) / (2 * (sigma**2)))

# Função para atualizar os pesos dos neurônios usando vizinhança gaussiana e decaimento da taxa de aprendizado
def update_weights(W, X, t, winner_idx, sigma, grid_size):
    """Atualiza os pesos dos neurônios da rede usando vizinhança gaussiana e taxa de aprendizado com decaimento."""
    eta = learning_rate_decay(t)  # Calcula a taxa de aprendizado decrescente
    for i in range(W.shape[0]):
        # Distância entre o neurônio atual e o vencedor no grid
        dist_to_winner = np.linalg.norm(np.array([i // grid_size, i % grid_size]) - np.array([winner_idx // grid_size, winner_idx % grid_size]))
        
        # Atualizar pesos com influência gaussiana
        influence = gaussian_neighbor(dist_to_winner, sigma)
        W[i, :] += influence * eta * (X - W[i, :])  # Aplica a fórmula de atualização com a nova taxa de aprendizado

# Inicializar a rede de Kohonen
def kohonen_train(X, grid_size=5, sigma_0=None, epochs=100):
    input_size = X.shape[1]
    W = np.random.rand(grid_size * grid_size, input_size)  # Inicialização aleatória dos pesos
    
    # Definir o valor de sigma_0 como o raio do mapa, se não fornecido
    if sigma_0 is None:
        sigma_0 = grid_size / 2
    
    # Calcular tau_1 conforme Haykin (2001)
    tau_1 = 1000 / math.log(sigma_0)
    
    prev_winner_indices = np.full(X.shape[0], -1)  # Inicializa com -1 para representar "sem vencedor anterior"

    for epoch in range(epochs):
        no_change = True  # Para verificar se houve mudança nos neurônios vencedores

        for i, x in enumerate(X):
            # Encontrar o neurônio vencedor
            distances = euclidean_distance(x, W)
            
            winner_idx = np.argmin(distances)
            
            # Verificar se houve mudança no neurônio vencedor
            if winner_idx != prev_winner_indices[i]:
                no_change = False  # Houve mudança
            prev_winner_indices[i] = winner_idx  # Atualizar o índice do neurônio vencedor


            # Calcular sigma(t) para a vizinhança gaussiana
            sigma_t = sigma_0 * np.exp(-epoch / tau_1)
            
            # Atualizar pesos com a vizinhança gaussiana e decaimento da taxa de aprendizado
            update_weights(W, x, epoch, winner_idx, sigma_t, grid_size)
        
        # Se não houve mudança nos neurônios vencedores, parar o treinamento
        if no_change:
            print(f'Treinamento interrompido no epoch {epoch}, pois não houve mudança no neurônio vencedor.')
            break      
    return W



def calculate_u_matrix(W, grid_size):
    u_matrix = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Índice do neurônio atual
            current_idx = i * grid_size + j
            distances = []

            # Verificar os vizinhos diretos (cima, baixo, esquerda, direita)
            if i > 0:  # Neurônio acima
                neighbor_idx = (i - 1) * grid_size + j
                distances.append(np.linalg.norm(W[current_idx] - W[neighbor_idx]))
            if i < grid_size - 1:  # Neurônio abaixo
                neighbor_idx = (i + 1) * grid_size + j
                distances.append(np.linalg.norm(W[current_idx] - W[neighbor_idx]))
            if j > 0:  # Neurônio à esquerda
                neighbor_idx = i * grid_size + (j - 1)
                distances.append(np.linalg.norm(W[current_idx] - W[neighbor_idx]))
            if j < grid_size - 1:  # Neurônio à direita
                neighbor_idx = i * grid_size + (j + 1)
                distances.append(np.linalg.norm(W[current_idx] - W[neighbor_idx]))

            # Média das distâncias para o neurônio atual
            u_matrix[i, j] = np.mean(distances) if distances else 0

    return u_matrix

def plot_all_u_matrices(u_matrices, titles):
    fig, axs = plt.subplots(1, len(u_matrices), figsize=(15, 5))

    for idx, u_matrix in enumerate(u_matrices):
        ax = axs[idx]
        im = ax.imshow(u_matrix, cmap='gray')
        ax.set_title(titles[idx])
        fig.colorbar(im, ax=ax)

    plt.show()

# Função para plotar os clusters e os centróides de cada grid em 3D
def plot_clusters_3d(W_list, grid_sizes, X2, d2):
    fig = plt.figure(figsize=(15, 5))
    
    # Definir cores para os clusters (as cores serão atribuídas manualmente)
    cluster_colors = ['blue', 'green', 'orange']  # Cores para os clusters
    cluster_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']  # Rótulos dos clusters
    
    # Iterar sobre as grades e dados
    for i, (W, grid_size) in enumerate(zip(W_list, grid_sizes)):
        ax = fig.add_subplot(1, len(W_list), i + 1, projection='3d')
        
        # Ajustar o K-means para a grade atual
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters_fit = kmeans.fit(W)
        clusters = clusters_fit.predict(W)

        # Testar os dados de teste em cada topologia usando os clusters K-Means
        y_pred = test_kmeans_on_kohonen(X2, W, clusters)

        # Mapear clusters para classes reais (Iris-setosa, Iris-versicolor, Iris-virginica)
        cluster_class_mapping = map_clusters_to_classes(y_pred, d2)

        # Agora, você pode usar essa mapeamento para renomear os clusters conforme as classes:
        y_pred = [cluster_class_mapping[cluster] for cluster in y_pred]

        # Calcular a acurácia para cada topologia
        accuracy = calculate_accuracy(y_pred, d2)
        
        # Mapear cada cluster a uma cor fixa
        cluster_color_map = [cluster_colors[label] for label in clusters]

        # Plotar os neurônios em 3D, coloridos de acordo com o cluster
        scatter = ax.scatter(W[:, 0], W[:, 1], W[:, 2], 
                             c=cluster_color_map, marker='o')
        
        # Plotar os centróides do K-means em 3D
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], 
                   color='red', marker='x', s=100, label='Centróides K-means')

        # Configurar título e legendas
        ax.set_title(f'Clusters para Grid {grid_size}x{grid_size}')
        ax.set_xlabel('SepalLength')
        ax.set_ylabel('SepalWidth')
        ax.set_zlabel('PetalLength')

        # Criar a legenda personalizada para os clusters
        for color, label in zip(cluster_colors, cluster_labels):
            ax.scatter([], [], color=color, label=label)  # Scatter vazio para legenda

        ax.legend(loc='upper right')
        ax.text2D(0.05, 0.95, f'Acurácia: {accuracy:.2f}%', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

# Função para testar a rede Kohonen com base nos clusters do K-Means
def test_kmeans_on_kohonen(X_test, W, clusters):
    y_pred = []
    for x in X_test:
        distances = euclidean_distance(x, W)
        winner_idx = np.argmin(distances)
        predicted_cluster = clusters[winner_idx]
        y_pred.append(predicted_cluster)
    return np.array(y_pred)

# Função para calcular a acurácia manualmente
def calculate_accuracy(y_pred, y_true):
    correct_predictions = np.sum(y_pred == y_true)
    total_predictions = len(y_true)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def map_clusters_to_classes(y_pred, y_true):
    # Mapear cada cluster para a classe mais comum
    cluster_class_map = {}
    for cluster_idx in range(3):  # Supondo 3 clusters
        # Pegar os índices dos pontos que pertencem a esse cluster
        points_in_cluster = np.where(y_pred == cluster_idx)[0]
        # Obter os rótulos verdadeiros desses pontos
        true_labels_in_cluster = y_true[points_in_cluster]
        # Contar a frequência de cada classe no cluster
        class_counts = Counter(true_labels_in_cluster)
        # Determinar a classe mais comum no cluster
        most_common_class = class_counts.most_common(1)[0][0]
        # Mapear o cluster à classe mais frequente
        cluster_class_map[cluster_idx] = most_common_class

    return cluster_class_map

# Ler os dados de treinamento e teste
X1, d1 = read_data('iris-tra.dat')
X2, d2 = read_data('iris-tst.dat')

# Normalizar os dados usando MinMaxScaler para o intervalo [0, 1]
scaler = MinMaxScaler()
X1 = scaler.fit_transform(X1)
X2 = scaler.fit_transform(X2)

# Definir parâmetros
grid_size = 5
epochs = 200
sigma_0 = grid_size / 2  # Valor inicial de sigma (raio do mapa)
tau_1 = 1000 / math.log(sigma_0)  # Constante de tempo, calculada com base em sigma_0

grid_size1 = 5
grid_size2 = 8
grid_size3 = 10

# Exemplo de uso após treinar cada topologia
W1 = kohonen_train(X1, grid_size1, sigma_0, epochs)
W2 = kohonen_train(X1, grid_size2, sigma_0, epochs)
W3 = kohonen_train(X1, grid_size3, sigma_0, epochs)

# Lista de todas as grades e tamanhos
W_list = [W1, W2, W3]
grid_sizes = [grid_size1, grid_size2, grid_size3]

# Plotar os clusters para cada grid
plot_clusters_3d(W_list, grid_sizes, X2, d2)

# Calcular a matriz-U para cada topologia
u_matrix1 = calculate_u_matrix(W1, grid_size1)
u_matrix2 = calculate_u_matrix(W2, grid_size2)
u_matrix3 = calculate_u_matrix(W3, grid_size3)

# Plotar as matrizes-U
plot_all_u_matrices([u_matrix1, u_matrix2, u_matrix3], [f'Grid {grid_size1}x{grid_size1}', f'Grid {grid_size2}x{grid_size2}', f'Grid {grid_size3}x{grid_size3}'])

