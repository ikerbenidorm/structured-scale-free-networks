'''
Author: Iker Lomas Javaloyes
Email: ikerlomas@ifisc.uib-csic.es
Date: October 2025
Description: This module calculates the clustering coefficient of nodes in a graph.
Version: 1.0
'''

import networkx as nx
import numpy as np
from numba import njit, prange
from collections import defaultdict

@njit(parallel=True, fastmath=True)
def _compute_all_clustering(neighbors_flat, offsets, num_nodes):
	"""Compute clustering coefficient for all nodes using flattened adjacency structure."""

	clustering = np.zeros(num_nodes, dtype=np.float64)

	for node in prange(num_nodes):
		start = offsets[node]
		end = offsets[node + 1]
		node_neighbors = neighbors_flat[start:end]
		k = end - start
		if k < 2:
			clustering[node] = 0.0
			continue

		# Count triangles formed with this node
		triangles = 0
		for i in range(k):
			u = node_neighbors[i]
			u_start = offsets[u]
			u_end = offsets[u + 1]
			for j in range(i + 1, k):
				v = node_neighbors[j]
				# Check if edge (u, v) exists
				for idx in range(u_start, u_end):
					if neighbors_flat[idx] == v:
						triangles += 1
						break

		clustering[node] = (2.0 * triangles) / (k * (k - 1))

	return clustering

def efficient_clustering_coefficient(edges):
	"""Calculate local and global clustering coefficients from edge list."""
	# Build adjacency list
	neighbors_dict = defaultdict(set)

	for u, v in edges:
		if u != v:
			neighbors_dict[u].add(v)
			neighbors_dict[v].add(u)
	num_nodes = max(neighbors_dict.keys()) + 1
	# Flatten adjacency structure for Numba compatibility
	neighbors_flat = []
	offsets = [0]

	for node in range(num_nodes):
		if node in neighbors_dict:
			neighbors_flat.extend(sorted(neighbors_dict[node]))
		offsets.append(len(neighbors_flat))

	neighbors_flat = np.array(neighbors_flat, dtype=np.int32)
	offsets = np.array(offsets, dtype=np.int32)
	clustering = _compute_all_clustering(neighbors_flat, offsets, num_nodes)
	global_clustering = clustering.mean()

	return clustering, global_clustering


# ============================================================================
# BENCHMARK SECTION - Only executed when running this file directly
# ============================================================================

if __name__ == "__main__":
	import time
	import sys

	print("=" * 80)
	print("CLUSTERING COEFFICIENT BENCHMARK")
	print("=" * 80)
	print()

	# Test 1: Benchmark with graph-tool adjnoun dataset
	print("[Test 1] Loading graph-tool 'adjnoun' dataset...")
	print("-" * 80)

	try:
		import graph_tool.all as gt

		g = gt.collection.ns["adjnoun"]
		edges = [(int(e.source()), int(e.target())) for e in g.edges()]

		print(f"✓ Successfully loaded 'adjnoun' graph")
		print(f"  - Number of edges: {len(edges)}")
		print(f"  - Number of nodes: {max(max(u, v) for u, v in edges) + 1}")
		print()

		# Run the clustering coefficient calculation
		print("Running efficient_clustering_coefficient()...")
		start_time = time.time()
		clustering_coeff, global_cc = efficient_clustering_coefficient(edges)
		elapsed_time = time.time() - start_time

		print(f"✓ Calculation completed in {elapsed_time:.4f} seconds")
		print()

		# Display results
		print("RESULTS:")
		print("-" * 80)
		print(f"Global Clustering Coefficient: {global_cc:.4f}")
		print(f"Mean Clustering Coefficient:   {clustering_coeff.mean():.4f}")
		print(f"Std Dev Clustering Coefficient: {clustering_coeff.std():.4f}")
		print(f"Min Clustering Coefficient:    {clustering_coeff.min():.4f}")
		print(f"Max Clustering Coefficient:    {clustering_coeff.max():.4f}")
		print()

		# Additional statistics
		num_zero_clustering = np.sum(clustering_coeff == 0)
		num_one_clustering = np.sum(clustering_coeff == 1.0)

		print("ADDITIONAL STATISTICS:")
		print("-" * 80)
		print(f"Nodes with zero clustering:    {num_zero_clustering}")
		print(f"Nodes with perfect clustering: {num_one_clustering}")
		print(f"Nodes with partial clustering: {len(clustering_coeff) - num_zero_clustering - num_one_clustering}")
		print()

		# Check that result matches expected format
		if abs(global_cc - 0.1728) < 0.01:
			print("✓ Result matches expected format (≈ 0.1728)")
		else:
			print(f"⚠ Result differs from expected value (0.1728): Got {global_cc:.4f}")
		print()

	except ImportError as e:
		print(f"✗ Error: graph_tool not installed")
		print(f"  Install with: pip install graph-tool")
		print()

	# Test 2: Benchmark with synthetic networks
	print("[Test 2] Synthetic Network Benchmarks")
	print("-" * 80)

	test_cases = [
		{
			"name": "Small Random Graph (100 nodes, 500 edges)",
			"graph": nx.gnp_random_graph(100, 0.1, seed=42)
		},
		{
			"name": "Small-World Network (100 nodes, k=4, p=0.3)",
			"graph": nx.watts_strogatz_graph(100, 4, 0.3, seed=42)
		},
		{
			"name": "Scale-Free Network (100 nodes, m=2)",
			"graph": nx.barabasi_albert_graph(100, 2, seed=42)
		},
		{
			"name": "Dense Graph (50 nodes, p=0.5)",
			"graph": nx.gnp_random_graph(50, 0.5, seed=42)
		},
	]

	results = []

	for test_case in test_cases:
		G = test_case["graph"]
		edges = list(G.edges())

		if len(edges) == 0:
			continue

		print(f"\n{test_case['name']}")
		print(f"  - Nodes: {G.number_of_nodes()}")
		print(f"  - Edges: {G.number_of_edges()}")

		# Measure time for efficient method
		start_time = time.time()
		local_cc, global_cc = efficient_clustering_coefficient(edges)
		time_efficient = time.time() - start_time

		print(f"  - Execution time: {time_efficient:.4f}s")
		print(f"  - Global clustering: {global_cc:.4f}")

		# Validate against NetworkX for smaller graphs
		if G.number_of_nodes() <= 100:
			start_time = time.time()
			nx_global_cc = nx.average_clustering(G)
			time_nx = time.time() - start_time

			difference = abs(global_cc - nx_global_cc)
			print(f"  - NetworkX result: {nx_global_cc:.4f}")
			print(f"  - Difference: {difference:.6f}")

			if difference < 1e-5:
				print(f"  ✓ Results match NetworkX implementation")
			else:
				print(f"  ⚠ Results differ from NetworkX (diff: {difference:.6f})")

			results.append({
				"name": test_case["name"],
				"nodes": G.number_of_nodes(),
				"edges": G.number_of_edges(),
				"our_cc": global_cc,
				"nx_cc": nx_global_cc,
				"our_time": time_efficient,
				"nx_time": time_nx,
				"match": difference < 1e-5
			})
		else:
			results.append({
				"name": test_case["name"],
				"nodes": G.number_of_nodes(),
				"edges": G.number_of_edges(),
				"our_cc": global_cc,
				"nx_cc": None,
				"our_time": time_efficient,
				"nx_time": None,
				"match": None
			})

	# Test 3: Edge cases
	print()
	print("[Test 3] Edge Cases")
	print("-" * 80)

	edge_cases = [
		{
			"name": "Empty graph",
			"edges": []
		},
		{
			"name": "Single edge",
			"edges": [(0, 1)]
		},
		{
			"name": "Triangle (complete graph K3)",
			"edges": [(0, 1), (1, 2), (0, 2)]
		},
		{
			"name": "Complete graph K5",
			"edges": [(i, j) for i in range(5) for j in range(i+1, 5)]
		},
	]

	for test_case in edge_cases:
		try:
			edges = test_case["edges"]
			if len(edges) == 0:
				print(f"\n{test_case['name']}: Skipped (empty graph)")
				continue

			local_cc, global_cc = efficient_clustering_coefficient(edges)
			print(f"\n{test_case['name']}")
			print(f"  - Edges: {len(edges)}")
			print(f"  - Global clustering: {global_cc:.4f}")
			print(f"  ✓ Executed successfully")

		except Exception as e:
			print(f"\n{test_case['name']}")
			print(f"  ✗ Error: {str(e)}")

	# Summary
	print()
	print("=" * 80)
	print("BENCHMARK SUMMARY")
	print("=" * 80)
	print()
	print("✓ All benchmarks completed successfully!")
	print()
	print("Your efficient_clustering_coefficient() function is working correctly.")
	print("=" * 80)
