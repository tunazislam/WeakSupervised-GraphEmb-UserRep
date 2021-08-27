# WeakSupervised-GraphEmb-UserRep

This repository contains code for [[Twitter User Representation using Weakly Supervised Graph Embedding]](https://arxiv.org/pdf/2108.08988.pdf), ICWSM 2022.

## Data:

1. Please download the 'data' folder from following link (on request):

[[Data]](https://purdue0-my.sharepoint.com/:f:/g/personal/islam32_purdue_edu/EnXigq9B9GxClDbCJR75DiQB-KrsmQRmplpSvKpU6QJNPQ?e=5vxwHu)

The dataset is parsed in an usable format for the codes in data.pickle. Files can be found at the following anonymous link:

2. 'data' folder should be kept inside the 'WeakSupervised-GraphEmb-UserRep/code' folder. 


## Computing Machine:

```
Supermicro SuperServer 7046GT-TR

Two Intel Xeon X5690 3.46GHz 6-core processors

48 GB of RAM

Linux operating system

Four NVIDIA GeForce GTX 1060 GPU cards for CUDA/OpenCL programming

```

## Software Packages and libraries:

```
python 3.6.6

PyTorch 1.1.0

jupiter notebook

pandas

pickle

gensim

nltk

nltk.tag

spacy

emoji

sklearn

statsmodels

scipy

matplotlib

numpy

preprocessor

transformers

```

## Code: 

### For Yoga:

1. Create graph embedding (Input files: weakly_all_des_gt_mergetweets_yoga_13k.csv, mention_yoga.pickle,  Output files: data_yoga_des_net.pickle, processed_yoga_data_des_net.pickle, yoga_graph_des_net.adjlist, yoga_graph_des_net.mapping) :

```
create_graph_des_@mention_weak_yoga_13k.ipynb 

```

2. For Embedding learning (Input files: data_yoga_des_net.pickle, Output files: b_model_des_net_yoga.m, predicted_utypes_des_net_yoga.pickle, user_des_net_yoga.embeddings): 


```

yoga_graph_des_@mention_embd.ipynb


```

3. For M step, run iteratively following codes: 


For first iteration, run following code

```

EM_yoga_graph_des_@mention_embd.ipynb

```

With Input files: processed_yoga_data_des_net.pickle, user_des_net_yoga.embeddings, weakly_all_des_gt_mergetweets_yoga_13k.csv, Output files: weakly_all_des_gt_mergetweets_yoga_13k_em1_des_net.csv


Then run following code:

```

EM_create_graph_des_@mention_weak_yoga_13k.ipynb

```

With Input files: weakly_all_des_gt_mergetweets_yoga_13k_em1_des_net.csv, mention_yoga.pickle,  Output files: data_yoga_des_net_em1.pickle, yoga_graph_des_net_em1.adjlist, yoga_graph_des_net_em1.mapping



For second iteration, run following code

```

EM_yoga_graph_des_@mention_embd.ipynb

```

With Input files: data_yoga_des_net_em1.pickle, processed_yoga_data_des_net.pickle, user_des_net_yoga_em1.embeddings, weakly_all_des_gt_mergetweets_yoga_13k_em1_des_net.csv, Output files: b_model_des_net_yoga_em1.m, predicted_utypes_des_net_yoga_em1.pickle, user_des_net_yoga_em1.embeddings, weakly_all_des_gt_mergetweets_yoga_13k_em2_des_net.csv


Then run following code:

```

EM_create_graph_des_@mention_weak_yoga_13k.ipynb

```

With Input files: weakly_all_des_gt_mergetweets_yoga_13k_em2_des_net.csv, mention_yoga.pickle,  Output files: data_yoga_des_net_em2.pickle, yoga_graph_des_net_em2.adjlist, yoga_graph_des_net_em2.mapping



For third iteration, run following code

```

EM_yoga_graph_des_@mention_embd.ipynb

```

With Input files: data_yoga_des_net_em2.pickle, processed_yoga_data_des_net.pickle, user_des_net_yoga_em2.embeddings, weakly_all_des_gt_mergetweets_yoga_13k_em2_des_net.csv, Output files: b_model_des_net_yoga_em2.m, predicted_utypes_des_net_yoga_em2.pickle, user_des_net_yoga_em2.embeddings, weakly_all_des_gt_mergetweets_yoga_13k_em3_des_net.csv


4. Supervised Baseline for yoga: 

   4.1) For LSTM_Glove, run 
	
	```
	baseline_glove_lstm_yoga_13k_weak.ipynb


	```

   4.2) For fine-tuned BERT, run 


	```
	baseline_BERT_finetuned_tweet_yoga_13k_weak.ipynb

	
	```

   For 3-fold cross-validation, randomly splitted input files are:
   For training: baseline_train_yoga_weak_cv1.csv, baseline_train_yoga_weak_cv2.csv, baseline_train_yoga_weak_cv3.csv
   For validation: baseline_val_yoga_weak_cv1.csv, baseline_val_yoga_weak_cv2.csv, baseline_val_yoga_weak_cv3.csv
   For testing: baseline_test_yoga_weak_cv1.csv, baseline_test_yoga_weak_cv2.csv, baseline_test_yoga_weak_cv3.csv
	

5. Visualize embeddings after label propagation: (input files: user_des_net_yoga.embeddings and predicted_yoga_13k_lp.csv )
	
	
	```
	visualize_graph_learning_embeddings_yoga.ipynb
	

	```
6. Visualize embeddings after EM: (input files: user_des_net_yoga_em2.embeddings and predicted_yoga_13k.csv )
	
	
	```
	visualize_embeddings_EM2_yoga.ipynb
	
	```



### For Keto:


1. Create graph embedding (Input files: weakly_all_des_mergetweets_keto_gt_14k.csv, mention_keto.pickle,  Output files: data_keto_des_net.pickle, processed_keto_data_des_net.pickle, keto_graph_des_net.adjlist, keto_graph_des_net.mapping) :

```
create_graph_des_@mention_weak_keto_14k.ipynb 

```

2. For Embedding learning (Input files: data_keto_des_net.pickle, Output files: b_model_des_net_keto.m, predicted_utypes_des_net_keto.pickle, user_des_net_keto.embeddings): 


```

keto_graph_des_@mention_embd.ipynb


```

3. For M step, run iteratively following codes: 


For first iteration, run following code

```

EM_keto_graph_des_@mention_embd.ipynb

```

With Input files: processed_keto_data_des_net.pickle, user_des_net_keto.embeddings, weakly_all_des_mergetweets_keto_gt_14k.csv, Output files: weakly_all_des_mergetweets_keto_gt_14k_em1_des_net.csv


Then run following code:

```

EM_create_graph_des_@mention_weak_keto_14k.ipynb

```

With Input files: weakly_all_des_mergetweets_keto_gt_14k_em1_des_net.csv, mention_keto.pickle,  Output files: data_keto_des_net_em1.pickle, keto_graph_des_net_em1.adjlist, keto_graph_des_net_em1.mapping



For second iteration, run following code

```

EM_keto_graph_des_@mention_embd.ipynb

```

With Input files: data_keto_des_net_em1.pickle, processed_keto_data_des_net.pickle, user_des_net_keto_em1.embeddings, weakly_all_des_mergetweets_keto_gt_14k_em1_des_net.csv, Output files: b_model_des_net_keto_em1.m, predicted_utypes_des_net_keto_em1.pickle, user_des_net_keto_em1.embeddings, weakly_all_des_mergetweets_keto_gt_14k_em2_des_net.csv


Then run following code:

```

EM_create_graph_des_@mention_weak_keto_14k.ipynb

```

With Input files: weakly_all_des_mergetweets_keto_gt_14k_em2_des_net.csv, mention_yoga.pickle,  Output files: data_keto_des_net_em2.pickle, keto_graph_des_net_em2.adjlist, keto_graph_des_net_em2.mapping



For third iteration, run following code

```

EM_keto_graph_des_@mention_embd.ipynb

```

With Input files: data_keto_des_net_em2.pickle, processed_keto_data_des_net.pickle, user_des_net_keto_em2.embeddings, weakly_all_des_mergetweets_keto_gt_14k_em2_des_net.csv, Output files: b_model_des_net_keto_em2.m, predicted_utypes_des_net_keto_em2.pickle, user_des_net_keto_em2.embeddings, weakly_all_des_mergetweets_keto_gt_14k_em3_des_net.csv


Then run following code:

```

EM_create_graph_des_@mention_weak_keto_14k.ipynb

```

With Input files: weakly_all_des_mergetweets_keto_gt_14k_em3_des_net.csv, mention_yoga.pickle,  Output files: data_keto_des_net_em3.pickle, keto_graph_des_net_em3.adjlist, keto_graph_des_net_em3.mapping


For fourth iteration, run following code

```

EM_keto_graph_des_@mention_embd.ipynb

```

With Input files: data_keto_des_net_em3.pickle, processed_keto_data_des_net.pickle, user_des_net_keto_em3.embeddings, weakly_all_des_mergetweets_keto_gt_14k_em3_des_net.csv, Output files: b_model_des_net_keto_em3.m, predicted_utypes_des_net_keto_em3.pickle, user_des_net_keto_em3.embeddings, weakly_all_des_mergetweets_keto_gt_14k_em4_des_net.csv




4. Supervised Baseline for keto:

   4.1) For LSTM_Glove, run 

	```
	baseline_glove_lstm_keto_14k_weak.ipynb

	```

   4.2) For fine-tuned BERT, run 

	```
	baseline_BERT_finetuned_tweet_keto_14k_weak.ipynb

	```

   For 3-fold cross-validation, randomly split input files are:
   For training: baseline_train_keto_weak_cv1.csv, baseline_train_keto_weak_cv2.csv, baseline_train_keto_weak_cv3.csv
   For validation: baseline_val_keto_weak_cv1.csv, baseline_val_keto_weak_cv2.csv, baseline_val_keto_weak_cv3.csv
   For testing: baseline_test_keto_weak_cv1.csv, baseline_test_keto_weak_cv2.csv, baseline_test_keto_weak_cv3.csv



5. Visualize embeddings after label propagation: (input files: user_des_net_keto.embeddings and predicted_keto_14k_lp.csv )
	
	
	```
	
	visualize_graph_learning_embeddings_keto.ipynb


	```
	

6. Visualize embeddings after EM: (input files: user_des_net_keto_em3.embeddings and predicted_keto_14k.csv )


	```
	
	visualize_embeddings_EM3_keto.ipynb


	```


