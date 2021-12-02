import pandas as pd
import os
pd.set_option('display.max_colwidth', None)

import transformers
import torch
import logging
import string
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from simpletransformers.t5 import T5Model, T5Args

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = Seq2SeqArgs()#Seq2SeqArgs()#T5Args()#S
model_args.n_gpu = 1
model_args.optimizer = "AdamW"
model_args.unlikelihood_loss = True
model_args.unlikelihood_loss_alpha_rank = 1
# model_args.dynamic_quantize = True
model_args.eval_batch_size = 2 #1
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 250
model_args.evaluate_during_training_verbose = True
model_args.save_optimizer_and_scheduler = False
model_args.evaluate_generated_text = True
model_args.save_best_model= True
model_args.early_stopping_metric = "diversity_score_mean"
model_args.early_stopping_metric_minimize = True
model_args.use_early_stopping= True
model_args.early_stopping_patience=5
model_args.fp16 = True
model_args.learning_rate = 4e-5#0.001#3e-4#3e-5#1e-4#
model_args.max_seq_length = 60
model_args.max_length = 60
model_args.num_train_epochs = 4
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.save_eval_checkpoints = True
model_args.save_model_every_epoch = True
model_args.save_steps = 250
model_args.train_batch_size = 4
model_args.use_multiprocessing = False
model_args.use_multiprocessed_decoding = False
model_args.warmup_steps = 100

model_args.EISLNatCriterion = True
model_args.eisl_ngram_factor = 0.2
model_args.ce_factor = 0.8
model_args.eisl_ngram = '2,3,4'

model_args.do_sample = True
model_args.num_beams = 10
model_args.top_k = 7
model_args.top_p = 0.99
model_args.temperature = 2.7
model_args.length_penalty = 0.5
model_args.num_return_sequences = 10
model_args.repetition_penalty = 2.2
model_args.max_length = 60

from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
import os
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.feature_extraction.text import CountVectorizer 
from itertools import combinations
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_cleaned_sentence(sentence):
    cleaned_sent = sentence.translate(str.maketrans('', '', string.punctuation))
    cleaned_sent = cleaned_sent.lower()
    return cleaned_sent

def is_same_words(original, rephrased):
    return get_cleaned_sentence(original) == get_cleaned_sentence(rephrased)
    
def remove_duplicates(sentence_to_rephrase, candidates):
    
    cleaned_outputs = set()
    final_sentences = set()
    for sentence in list(candidates):
        # Only add unique sentences (remove punctuation to determine uniquness)
        cleaned_sent = get_cleaned_sentence(sentence)
        if(cleaned_sent not in cleaned_outputs):
            final_sentences.add(sentence)
            cleaned_outputs.add(cleaned_sent)

    return list(final_sentences)

def vect_similarity(a,b):
    vectorizer = CountVectorizer(ngram_range=(1,3))

    vecs = vectorizer.fit_transform([a,b])
    return cosine_similarity(vecs[0], vecs[1])[0]

def gleu_score(labels, preds, ismax=False,isdiversity=False):
    prediction_df = val_data.merge(pd.DataFrame(list(zip(labels, preds)),columns=["target_text","preds"]),on="target_text")
    prediction_df = prediction_df[~prediction_df['target_text'].isin(['acceptable','not acceptable'])]
    gleu_scores = []
    gleu_scores_max = []
    diversity_scores = []
    diversity_scores_reference = []
    for sentence_to_rephrase, references in prediction_df.groupby("input_text"):
        cur_scores_gleu = []
        diversity_scores_sim= []
        predictions = list(set(np.array(references.preds.values[0]).flatten().tolist()))
        predictions = remove_duplicates(sentence_to_rephrase,predictions)
        
        tokenized_references = [nltk.word_tokenize(reference) for reference in references.target_text.values]
        
        for sentence in predictions:
            gleu = sentence_gleu(tokenized_references, nltk.word_tokenize(sentence))
            cur_scores_gleu.append(gleu)
            
        if len(predictions) > 1:
            for couple in combinations(predictions, 2):
                sim = vect_similarity(couple[0], couple[1])
                diversity_scores_sim.append(sim)
        
        diversity_scores_reference.append(np.mean([vect_similarity(couple[0], couple[1]) for couple in combinations(references.target_text.values, 2)]))
        
        if len(diversity_scores_sim) > 0:
            diversity_scores.append(np.mean(diversity_scores_sim))
        gleu_scores.append(np.mean(cur_scores_gleu))
        gleu_scores_max.append(np.max(cur_scores_gleu))
        
        if isdiversity:
            num_of_results = min(len(cur_scores_gleu),3)
            best = np.argpartition(cur_scores_gleu, -1*num_of_results)[-1*num_of_results:]
            for i in best:
                print("  ",predictions[i],cur_scores_gleu[i])
            print("---")
                
    if isdiversity:
        return np.mean(diversity_scores)
    else:
        if ismax:
            print("Gleu Mean: {:.02f}\nGleu Max: {:.02f}\nDiversity Score: {:.02f}\nDiversity Score (Reference): {:.02f}\n".format(
              np.mean(gleu_scores), np.mean(gleu_scores_max), np.mean(diversity_scores), np.mean(diversity_scores_reference)))
            return np.mean(gleu_scores_max)
        else:
            return np.mean(gleu_scores)

def gleu_score_max(labels, preds):
    return gleu_score(labels,preds, True)

def gleu_score_mean(labels, preds):
    return gleu_score(labels,preds, False)

def diversity_score_mean(labels, preds):
    return gleu_score(labels,preds, False, True)

def read_data(path):
    data = pd.read_csv(path, sep="\\t", engine='python')
    data.rename(columns={'original': 'input_text', 'suggestions': 'target_text'}, inplace=True)

    return data

df = pd.read_csv("results_up_to_23_11.csv", sep='\t').drop_duplicates()
df = df[["input_sentence", "rephraser_result", "rating"]]
df = df.rename(columns={"input_sentence": "input_text", "rephraser_result": "target_text"})
df = df[df.rating != 3]
df.loc[df.rating >= 4, "target_text"] = "1 " + df["target_text"]
df.loc[df.rating <= 2, "target_text"] = "0 " + df["target_text"]
# df = df.sample(4)
# df = df[df.rating <=2].sample(4)
# print(df)

val_data = read_data("validation.csv")

model = Seq2SeqModel(encoder_decoder_type="bart", encoder_decoder_name="facebook/bart-base", args=model_args, use_cuda=False)
model.train_model(df.dropna(subset=["input_text","target_text"]), eval_data=val_data.dropna(),gleu_score_max=gleu_score_max, gleu_score_mean=gleu_score_mean,diversity_score_mean=diversity_score_mean)



