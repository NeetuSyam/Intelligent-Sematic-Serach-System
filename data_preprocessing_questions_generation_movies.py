import pandas as pd
import time
from tqdm import tqdm
import seaborn as sns
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration


class QueryGenerator:
    """Class for generating queries using T5 model."""

    def __init__(self, data_path, model_name='BeIR/query-gen-msmarco-t5-large-v1', device='cuda'):
        """
        Initialize the QueryGenerator.

        Args:
            data_path (str): Path to the data file.
            model_name (str): Name of the T5 model to use.
            device (str): Device to run the model on.
        """
        self.data_path = data_path
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.data_df = None

    def load_data(self):
        """Load and preprocess the data."""
        data = pd.read_csv(self.data_path)
        df = data[['Title', 'Plot']]
        del data
        df.dropna(inplace=True)
        df.drop_duplicates(subset=['Plot'], inplace=True)
        self.data_df = df

    def initialize_model(self):
        """Initialize the tokenizer and model."""
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def generate_queries(self, output_path, batch_size=16, num_queries=5, max_length_paragraph=512, max_length_query=64):
        """
        Generate queries for paragraphs and save them to a file.

        Args:
            output_path (str): Path to save the generated queries.
            batch_size (int): Batch size for generating queries.
            num_queries (int): Number of queries to generate per paragraph.
            max_length_paragraph (int): Maximum length of the input paragraphs.
            max_length_query (int): Maximum length of the generated queries.
        """
        paragraphs = self.data_df.Plot.tolist()

        def _remove_non_ascii(s):
            """Remove non-ASCII characters from a string."""
            return "".join(i for i in s if ord(i) < 128)

        # Generate queries for paragraphs and save to file
        with open(output_path, 'w') as fOut:
            for start_idx in tqdm(range(0, len(paragraphs), batch_size)):
                sub_paragraphs = paragraphs[start_idx:start_idx + batch_size]
                inputs = self.tokenizer.prepare_seq2seq_batch(sub_paragraphs, max_length=max_length_paragraph,
                                                              truncation=True,
                                                              return_tensors='pt').to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length_query,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=num_queries)

                for idx, out in enumerate(outputs):
                    query = self.tokenizer.decode(out, skip_special_tokens=True)
                    query = _remove_non_ascii(query)
                    para = sub_paragraphs[int(idx / num_queries)]
                    para = _remove_non_ascii(para)
                    fOut.write("{}\t{}\n".format(query.replace("\t", " ").strip(), para.replace("\t", " ").strip()))

    def run(self, output_path, batch_size=16, num_queries=5, max_length_paragraph=512, max_length_query=64):
        """
        Run the query generation process.

        Args:
            output_path (str): Path to save the generated queries.
            batch_size (int): Batch size for generating queries.
            num_queries (int): Number of queries to generate per paragraph.
            max_length_paragraph (int): Maximum length of the input paragraphs.
            max_length_query (int): Maximum length of the generated queries.
        """
        self.load_data()
        self.initialize_model()
        self.generate_queries(output_path, batch_size, num_queries, max_length_paragraph, max_length_query)


# Example usage
data_path = 'wiki_movie_plots_deduped.csv'
output_path = 'generated_queries_all_new_movie.tsv'

query_generator = QueryGenerator(data_path)
query_generator.run(output_path)
