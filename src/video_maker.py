import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
sys.path.append(script_dir)

import numpy as np
from HighClusteredModel import hc_scale_free_graph
from plots_networks import plot_network_simple
from clustering_coefficient import efficient_clustering_coefficient
import imageio
from PIL import Image
from multiprocessing import Pool, cpu_count
import subprocess

folder = '../data/temp_frames'
os.makedirs(folder, exist_ok=True)

config1_files = []
config2_files = []

def process_single_node(args):
    """Porcess unique value of nodes"""
    i, l = args
    Graph1 = hc_scale_free_graph(l, m=10, a=10, directed=False, seed=43254)
    Graph2 = hc_scale_free_graph(l, m=2, a=2, directed=False, seed=70982)
    
    ccl1 = efficient_clustering_coefficient(Graph1.edges())[-1].item()
    ccl2 = efficient_clustering_coefficient(Graph2.edges())[-1].item()

    file1 = f'{folder}/config1_frame_{i:03d}.png'
    file2 = f'{folder}/config2_frame_{i:03d}.png'

    plot_network_simple(Graph1, file1, l, ccl1, "Config 1 (m=10, a=10)")
    plot_network_simple(Graph2, file2, l, ccl2, "Config 2 (m=2, a=2)")

    return file1, file2

nods = list(map(int, np.logspace(1, 4, num=1000)))
num_cores = max(1, cpu_count())

with Pool(processes=num_cores) as pool:
    results = pool.map(process_single_node, enumerate(nods))

for file1, file2 in results:
    config1_files.append(file1)
    config2_files.append(file2)
    

# Reordenate the files by frame
config1_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
config2_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
with imageio.get_writer(f'../data/network_comparison.mp4', fps=10) as writer:
    for i, (file1, file2) in enumerate(zip(config1_files, config2_files)):
        # Create composite image
        img1 = Image.open(file1)
        img2 = Image.open(file2)

        # Create new image with double the width
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)

        new_img = Image.new('RGB', (total_width, max_height), 'white')
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))

        # Convert to numpy array for imageio
        frame = np.array(new_img)
        writer.append_data(frame)
        
subprocess.run(['rm', '-rf', folder])