import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt


#Generating the  ER Graph
def generate_er_graph(n, p):
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            edge = np.random.rand() < p
            A[i, j] = A[j, i] = int(edge)
    return A

def expected_edges(n, p):
    E = p * n * (n - 1) / 2
    Var = p * (1 - p) * n * (n - 1) / 2
    return E, Var

def visualize_er_graph(array):
    G = nx.from_numpy_array(array)
    nx.draw(G, with_labels=True, node_color='blue', edge_color='green')
    plt.show()

array = generate_er_graph(10, 0.3)
visualize_er_graph(array)

#Converting adjacency to sequence
def adjacency_to_sequence(A):
    n = A.shape[0]
    sequence = []
    for i in range(n):
        for j in range(i + 1, n):
            sequence.append(A[i, j])
    return sequence

#Arithmetic Coding algorithm
def arithmetic_compress(sequence, p):
    """
    Simple arithmetic coding for Bernoulli source.
    sequence: list of 0/1 bits
    p: probability of 1
    """
    low, high = 0.0, 1.0
    for bit in sequence:
        range_ = high - low
        if bit == 1:
            high = low + range_ * p
        else:
            low = low + range_ * p
    # Final code is any number in [low, high)
    code = (low + high) / 2
    # Approximate compressed length in bits
    compressed_length = -math.log2(high - low)
    return code, compressed_length

# Compression Ratio and Entropy Bound
def compression_ratio(original_bits, compressed_length):
    return len(original_bits) / compressed_length

def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def run_experiment(n, p):
    A = generate_er_graph(n, p)
    sequence = adjacency_to_sequence(A)
    original_bits = ''.join(map(str, sequence))
    code, compressed_length = arithmetic_compress(sequence, p)
    R = compression_ratio(original_bits, compressed_length)
    H = entropy(p)
    E, Var = expected_edges(n, p)
    print(f"Number of vertices (n): {n}")
    print(f"Edge probability (p): {p}")
    print(f"Expected number of edges: {E:.2f}")
    print(f"Variance of edges: {Var:.2f}")
    print(f"Original size (bits): {len(original_bits)}")
    print(f"Compressed size (approx bits): {compressed_length:.2f}")
    print(f"Compression Ratio R: {R:.4f}")
    print(f"Theoretical Entropy Bound H(p): {H:.4f}")
    print(f"Arithmetic Code (representative value): {code}")

if __name__ == "__main__":
    run_experiment(n=10, p=0.3)
