import numpy as np

def print_statistics(loader):
    # Takes a pytorch dataloader and prints out some statistics about the dataset
    lengths = [data['graph'].x.shape[0] for data in loader.dataset]
    
    max_length = np.max(lengths)
    min_length = np.min(lengths)
    avg_length = np.mean(lengths)
    median_length = np.median(lengths)

    for batch in loader:
        graph = batch['graph']
        print(f'Batch node features shape: \t{graph.x.shape}')
        print(f'Batch edge index shape: \t{graph.edge_index.shape}')
        print(f'Batch labels shape: \t\t{graph.y.shape}')
        break

    
    print("Max Length:", max_length)
    print("Min Length:", min_length)
    print("Average Length:", avg_length)
    print("Median Length:", median_length)

    return {
        "max_length": max_length,
        "min_length": min_length,
        "average_length": avg_length,
        "median_length": median_length
    }