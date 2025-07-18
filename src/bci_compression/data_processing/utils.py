"""
Utility functions for data processing.
"""
def flatten(lst):
    return [item for sublist in lst for item in sublist]

def chunk_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]
