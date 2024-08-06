# MSCI641-Final-Project
Team member: Jie Liu, Yijin Yin

In Task 2, we utilized the Hugging Face Transformers library to implement the spoiler generation model. The specific implementation steps included:

Loading the pre-trained DistilBERT model and tokenizer:
Code path: src/transformers/models/distilbert/tokenization_distilbert.py for the DistilBertTokenizer class.
Code path: src/transformers/models/distilbert/modeling_distilbert.py for the DistilBertForMaskedLM class.
Optimizer:
Code path: src/transformers/optimization.py for the AdamW class. 

Github by https://github.com/huggingface/transformers
