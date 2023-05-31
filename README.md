# Intelligent-Sematic-Serach-System

# Steps to Run the Project

To install required packages
```
!pip install faiss-gpu
!pip install sentence-transformers
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding
```

For generating questions for **Movies** dataset
[_change the datapath after def run()_]
```
!python data_preprocessing_questions_generation_movies.py 
```

For generating questions for **News** dataset
[_change the datapath after def run()_]

```
!python data_preprocess_questions_news.py
```

For generating questions for **Arxiv** dataset
[_change the datapath after def run()_]
```
!python preprocess_question_arxiv.py
```

Finetune **Movies** dataset with base distill bert and **sentence-transformers/msmarco-distilbert-cos-v5**
```
!python movie_finetuning_ditill_dot.py /content/generated_queries_all_mo.tsv 'sentence-transformers/msmarco-distilbert-base-tas-b'
```

Finetune **News** dataset with base distill bert and later with **sentence-transformers/msmarco-distilbert-dot-v5**
```
!python stack_finetuning.py --train_data_paths /content/generated_queries_all_mo.tsv /content/generated_queries_all_arxiv.tsv --model_name 'sentence-transformers/msmarco-distilbert-base-tas-b'
```

Finetune **Stack** dataset with  **sentence-transformers/msmarco-distilbert-cos-v5**
```
!python finetuning_stack --train_data_paths /content/generated_queries_all_mo.tsv /content/drive/MyDrive/data240/generated_queries_all_arxiv.tsv --model_name sentence-transformers/msmarco-distilbert-cos-v5
```

Finding accuracy for **Movie** dataset with base distill bert finetuned model with dot product. [_you can change data and model path_]
```
!python accuracy_movies.py --question_file /content/drive/MyDrive/data240/train_gen_q_mov.csv --dataset_file /content/drive/MyDrive/data240/wiki_movie_plots_deduped.csv --model_path /content/drive/MyDrive/data240/base-distillbert-movie --similarity dot --output_file output.txt
```

Finding accuracy for **Movie** dataset with base distill bert finetuned model with cosine similarity. [_you can change data and model path_]
```
!python accuracy_movies.py --question_file /content/drive/MyDrive/data240/train_gen_q_mov.csv --dataset_file /content/drive/MyDrive/data240/wiki_movie_plots_deduped.csv --model_path /content/drive/MyDrive/data240/base-distillbert-movie --similarity cosine --output_file output.txt
```

Finding accuracy for **Movie** dataset with base distill bert finetuned model with cosine similarity. [_you can change data and model path_]
```
!python accuracy_news.py --question_file /content/drive/MyDrive/data240/train_qen_q_ag.csv --dataset_file /content/drive/MyDrive/data240/subsampled_dataset.csv --model_path /content/drive/MyDrive/data240/base-distillbert-movie --similarity dot --output_file output.txt
```

Finding accuracy for **News** dataset with base distill bert finetuned model with cosine similarity. [_you can change data and model path_]
```
!python accuracy_news.py --question_file /content/drive/MyDrive/data240/train_qen_q_ag.csv --dataset_file /content/drive/MyDrive/data240/subsampled_dataset.csv --model_path /content/drive/MyDrive/data240/base-distillbert-movie --similarity cosine --output_file output.txt
```

Finding accuracy for **News** dataset with msmarco-distilbert-dot-v5 finetuned model with dot product. [_you can change data and model path_]
```
!python accuracy_news.py --question_file /content/drive/MyDrive/data240/train_qen_q_ag.csv --dataset_file /content/drive/MyDrive/data240/subsampled_dataset.csv --model_path /content/drive/MyDrive/data240/finetune_news_dot --similarity dot --output_file output.txt
```

Finding accuracy for **Movie** dataset with msmarco-distilbert-cos-v5 finetuned model with cosine similarity. [_you can change data and model path_] 
```
!python accuracy_movies.py --question_file /content/drive/MyDrive/data240/train_gen_q_mov.csv --dataset_file /content/drive/MyDrive/data240/wiki_movie_plots_deduped.csv --model_path /content/drive/MyDrive/data240/finetune_movie_cos --similarity cosine --output_file output.txt
```

Finding accuracy for **Stack** dataset with msmarco-distilbert-cos-v5 finetuned model with cosine similarity. [_you can change data and model path_] 
```
!python accuracy_movies.py --question_file /content/drive/MyDrive/data240/train_gen_q_mov.csv --dataset_file /content/drive/MyDrive/data240/stack.csv --model_path /content/drive/MyDrive/data240/finetuning_stack --similarity cosine --output_file output.txt
```

Comparing execution of our own implementation of NN and faiss. [_Need to run twice beacause for the first time it takes time to initialize_]
```
!python flickr.py --data_path "/content/drive/MyDrive/data240/combined_comments.csv" --output_path "output.txt"
```

[News Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

[Flickr Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

[Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)

[Arxiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)

- Running preprocessing needs lot of computation (at least 24gb of vRAM)
- But running from accuracy can be done with less compuatation
- It is advised to run files with colab(A100 gpu), can follow the folowup.ipynb

[Datalink](https://drive.google.com/drive/folders/1-5FlM3FZfqlcKZ80Mas09RozJuQsdkrF?usp=sharing)

