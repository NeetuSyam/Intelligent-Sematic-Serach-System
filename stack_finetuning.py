import argparse
import os
import random
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets


class QueryRankingTrainer:
    """
    A class to train a query ranking model using Sentence Transformers.
    """
    def __init__(self, train_data_paths, model_name):
        """
        Initialize the QueryRankingTrainer class with the given parameters.

        Parameters:
        train_data_paths (list of str): List of paths to the training data files.
        model_name (str): Name of the Sentence Transformers model to use.
        """
        self.train_data_paths = train_data_paths
        self.model_name = model_name

    def load_train_data(self):
        """
        Load training data from the files and returns a list of InputExamples.

        Returns:
        List[InputExample]: List of InputExamples created from the training data.
        """
        train_examples = []
        for train_data_path in self.train_data_paths:
            with open(train_data_path) as fIn:
                for line in fIn:
                    try:
                        query, paragraph = line.strip().split('\t', maxsplit=1)
                        train_examples.append(InputExample(texts=[query, paragraph]))
                    except:
                        pass
        random.shuffle(train_examples)
        return train_examples

    def train_model(self, train_examples):
        """
        Train the model with the provided training examples.

        Parameters:
        train_examples (List[InputExample]): List of InputExamples for training the model.
        """
        train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=8)
        word_emb = models.Transformer(self.model_name)
        pooling = models.Pooling(word_emb.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_emb, pooling])

        train_loss = losses.MultipleNegativesRankingLoss(model)
        num_epochs = 3
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps, show_progress_bar=True)

        return model

    def save_model(self, model):
        """
        Save the trained model to the disk.

        Parameters:
        model (SentenceTransformer): The trained SentenceTransformer model to save.
        """
        model.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query Ranking Model Training')
    parser.add_argument('--train_data_paths', nargs='+', required=True, help='Paths to the training data files')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the Sentence Transformers model to use')

    args = parser.parse_args()

    trainer = QueryRankingTrainer(args.train_data_paths, args.model_name)
    train_examples = trainer.load_train_data()
    model = trainer.train_model(train_examples)
    trainer.save_model(model)
