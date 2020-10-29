import os
import numpy as np

class_id_dic = {
    'Case_Based': 0,
    'Genetic_Algorithms': 1,
    'Neural_Networks': 2,
    'Probabilistic_Methods': 3,
    'Reinforcement_Learning': 4,
    'Rule_Learning': 5,
    'Theory': 6
}


def load_cora(content, cites):
    with open(content, 'r') as feat_file:
        features = []  # Nodes features
        indices = []  # paper_id
        labels = []  # class label
        # We can't really do better than reading the whole file line by line
        for line in feat_file:
            line_content = line.split()
            indices.append(line_content[0])
            labels.append(class_id_dic[line_content[-1]])
            features.append(list(map(int, line_content[1:-1])))
    # Convert to numpy array once and for all
    features = np.array(features)
    labels = np.array(labels)

    # Build -symetric- adjacency matrix
    # The links are directed in the dataset but the paper treats undirected graphs
    nb_nodes = labels.size
    adjacency = np.zeros((nb_nodes, nb_nodes))
    with open(cites, 'r') as adj_file:
        for line in adj_file:
            id1, id2 = line.split()
            idx1, idx2 = indices.index(id1), indices.index(id2)
            # Construct a binary matrix, as in the paper (some edges could have been counted twice)
            adjacency[idx1, idx2] = 1
            adjacency[idx2, idx1] = 1
    return (features, adjacency, labels)


if __name__ == '__main__':
    path = os.getcwd() + '/cora_own/'
    features, adjacency, labels = load_cora(path+'cora.content', path+'cora.cites')
    print('Shapes:', features.shape, adjacency.shape)
    print('Avj words per doc:', np.mean(np.sum(features, axis=1)))
    print('Nb of citations (x2):', np.sum(adjacency))
