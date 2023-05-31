import pandas as pd
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import cupy as cp

class ImageComparison:
    """
    A class used to compare image search results using FAISS and CUDA.
    """

    def __init__(self, data_file, output_file):
        """
        Initializes the class with a data file and an output file.
        
        Parameters:
        data_file (str): The path to the data file.
        output_file (str): The path to the output file where results will be written.
        """
        self.df = pd.read_csv(data_file)
        self.df.dropna(inplace=True)
        self.model = SentenceTransformer('msmarco-distilbert-cos-v5', device='cuda')
        self.encoded_data = self.model.encode(self.df[' comment'].tolist())
        self.encoded_data = np.asarray(self.encoded_data.astype('float32'))
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        faiss.normalize_L2(self.encoded_data)
        self.index.add_with_ids(self.encoded_data, np.array(range(0, len(self.df))))
        self.output_file = output_file

    def fetch_info(self, dataframe_idx):
        """
        Fetches image information from the data frame.

        Parameters:
        dataframe_idx (int): The index of the image in the data frame.
        
        Returns:
        dict: A dictionary with the image name and comment.
        """
        info = self.df.iloc[dataframe_idx]
        meta_dict = {}
        meta_dict['image_name'] = info['image_name']
        meta_dict[' comment'] = info[' comment']
        return meta_dict

    def search_faiss(self, query, top_k):
        """
        Searches for images using FAISS.

        Parameters:
        query (str): The search query.
        top_k (int): The number of results to return.
        
        Returns:
        float: The time taken for the search.
        """
        start_time = time.time()
        query_vector = self.model.encode([query])
        top_k = self.index.search(query_vector, top_k)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time

    def nearest_neighbors_cuda(self, query, k):
        """
        Searches for images using CUDA.

        Parameters:
        query (str): The search query.
        k (int): The number of results to return.
        
        Returns:
        float: The time taken for the search.
        """
        query_vector = self.model.encode(query)
        query_vector = cp.asarray(query_vector)
        encoded_data = cp.asarray(self.encoded_data)
        query_norm = cp.linalg.norm(query_vector)
        data_norm = cp.linalg.norm(encoded_data, axis=1)
        cosine_distances = 1.0 - cp.dot(encoded_data, query_vector) / (data_norm * query_norm)
        start_time = time.time()
        nearest_indices = cp.argsort(cosine_distances)[:k]
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time

    def compare_methods(self, query, top_k):
        """
        Compares the time taken for a search using FAISS and CUDA and writes the results to a file.

        Parameters:
        query (str): The search query.
        top_k (int): The number of results to return.
        """
        time_faiss = self.search_faiss(query, top_k)
        time_cuda = self.nearest_neighbors_cuda(query, top_k)
        with open(self.output_file, 'a') as f:
            f.write(f"Time taken by FAISS: {time_faiss}\n")
            f.write(f"Time taken by CUDA: {time_cuda}\n")

if __name__ == "__main__":
    comparison = ImageComparison("/content/drive/MyDrive/data240/combined_comments.csv", "output.txt")
    comparison.compare_methods("psycho thriller", 10)
    
