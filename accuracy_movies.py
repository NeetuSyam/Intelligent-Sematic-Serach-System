import pandas as pd
import time
from tqdm import tqdm
import seaborn as sns
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss
import argparse


class AccuracyEvaluator:
    def __init__(self, question_file, dataset, model_path, similarity):
        self.question_file = question_file
        self.dataset = dataset
        self.model_path = model_path
        self.similarity = similarity

        self.model = None
        self.index = None
        self.df = None
        self.df2 = None

    def load_data(self):
        # Load question data
        self.df = pd.read_csv(self.question_file)
        self.df = self.df.dropna(subset=['1'])
        self.df = self.df.reset_index(drop=True)

        # Load dataset
        self.df2 = pd.read_csv(self.dataset)
        self.df2 = self.df2[['Title', 'Plot']]

    def load_model(self):
        # Load SentenceTransformer model
        self.model = SentenceTransformer(self.model_path, device='cuda')

    def encode_data(self):
        # Encode data using SentenceTransformer model
        encoded_data_st = self.model.encode(self.df2.Plot.tolist())
        encoded_data = np.asarray(encoded_data_st.astype('float32'))

        if self.similarity == "cosine":
            # Create Faiss index with cosine similarity
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        elif self.similarity == "dot":
            # Create Faiss index with dot similarity
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
            faiss.normalize_L2(encoded_data)
        else:
            raise ValueError("Invalid similarity option. Choose either 'cosine' or 'dot'.")

        self.index.add_with_ids(encoded_data, np.array(range(0, len(self.df2))))
        faiss.write_index(self.index, 'news.index')

    @staticmethod
    def fetch_stack_info(dataframe_idx, df2):
        info = df2.iloc[dataframe_idx]
        meta_dict = dict()
        meta_dict['Title'] = info['Title']
        meta_dict['Plot'] = info['Plot'][:500]
        return meta_dict

    def search_stack(self, query, top_k):
        query_vector = self.model.encode([query])
        top_k = self.index.search(query_vector, top_k)
        top_k_ids = top_k[1].tolist()[0]
        top_k_ids = list(np.unique(top_k_ids))
        results = [self.fetch_stack_info(idx, self.df2) for idx in top_k_ids]
        return results

    def calculate_accuracy(self, total_queries, top_k):
        correct_count = 0

        for i in range(total_queries):
            query = self.df['0'][i]
            description = self.df['1'][i]

            results = self.search_stack(query, top_k=top_k)

            for result in results:
                if description in result.get('Plot'):
                    correct_count += 1
                    break

        accuracy = correct_count / total_queries
        return accuracy


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Accuracy Evaluator')

    # Add arguments
    parser.add_argument('--question_file', type=str, help='Path to the question data file')
    parser.add_argument('--dataset_file', type=str, help='Path to the dataset file')
    parser.add_argument('--model_path', type=str, help='Path to the SentenceTransformer model')
    parser.add_argument('--similarity', type=str, help='Similarity type: cosine or dot')
    parser.add_argument('--output_file', type=str, help='Path to the output file')

    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of the AccuracyEvaluator
    evaluator = AccuracyEvaluator(args.question_file, args.dataset_file, args.model_path, args.similarity)
    evaluator.load_data()
    evaluator.load_model()
    evaluator.encode_data()

    total_queries = 1000  # Number of queries to evaluate
    top_k = 10  # Top-k results to consider

    accuracy = evaluator.calculate_accuracy(total_queries, top_k)

    # Write accuracy to the output file
    with open(args.output_file, 'a') as f:
        f.write(f"Model Path: {args.model_path}, Dataset: {args.dataset_file}, Similarity: {args.similarity}, Accuracy: {accuracy}\n")
