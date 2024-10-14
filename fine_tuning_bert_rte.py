import pandas as pd
import numpy as np

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

# Data download and preprocessing
df = pd.read_csv("your_dataset.csv")

from sklearn.model_selection import train_test_split

df_train = df_train[["label_text", "text"]] # Suggested names for dataset columns
df_test = df_test[["label_text", "text"]]

# Creating RTE hypothesis for vaccine-related headlines
hypothesis_label_dic = {
"global access to vaccine": "The headline addresses initiatives for equal access to covid-19 vaccines, such as COVAX, and/or the need to combat inequity, ensure global distribution, and contain the disease, especially to low- and middle-income countries.",
"science and technology": "The headline addresses the science behind vaccine development, including research, studies, safety and efficacy trials, emergency approval, and side effects.",
"public health policies": "The headline addresses vaccine mandates, vaccine passports, and/or other public health policies such as social distancing, quarantine, lockdowns, and mask mandates.",
"vaccination rollout/campaign": "The headline addresses vaccination rollout, campaigns, priority groups, vaccine deals, and/or the monitoring of vaccination rates.",
"vaccine hesitancy and mis-/disinformation": "The headline addresses the circulation of disinformation and misinformation regarding vaccines, which contribute to vaccine hesitancy and reluctance.",
"public endorsement to vaccine": "The headline addresses public endorsement and incentive for vaccines, such as celebrities and artistis getting vaccinated.",
"institutional affairs": "The headline addresses institutional and governmental issues, political disputes, conflicts, political authorities, and export bans.",
"problems in vaccination": "The headline addresses problems in vaccination, such as delays, fraud, disparities, vaccine shortages.",
"public perception of vaccine": "The headline addresses opinion polls and surveys on the public perception of vaccines and/or willingness to get vaccinated.",
"economic consequences": "The headline addresses the impacts and benefits of vaccination on the economy."
}

# Formatting the training and testing datasets for RTE classification

# Training dataset
def format_rte_trainset(df_train=None, hypo_label_dic=None, random_seed=42):
  length_original_data_train = len(df_train)

  df_train_lst = []
  for label_text, hypothesis in hypo_label_dic.items():
    # Entailment
    df_train_step = df_train[df_train.label_text == label_text].copy(deep=True)
    df_train_step["hypothesis"] = [hypothesis] * len(df_train_step)
    df_train_step["label"] = [0] * len(df_train_step)
    # Not entailment
    df_train_step_not_entail = df_train[df_train.label_text != label_text].copy(deep=True)
    df_train_step_not_entail = df_train_step_not_entail.sample(n=min(len(df_train_step), len(df_train_step_not_entail)), random_state=random_seed)
    df_train_step_not_entail["hypothesis"] = [hypothesis] * len(df_train_step_not_entail)
    df_train_step_not_entail["label"] = [1] * len(df_train_step_not_entail)
    # Append
    df_train_lst.append(pd.concat([df_train_step, df_train_step_not_entail]))
  df_train = pd.concat(df_train_lst)

  # Shuffle
  df_train = df_train.sample(frac=1, random_state=random_seed)
  df_train["label"] = df_train.label.apply(int)
  df_train["label_rte_explicit"] = ["True" if label == 0 else "Not-True" for label in df_train["label"]]

  return df_train.copy(deep=True)

df_train_formatted = format_rte_trainset(df_train=df_train, hypo_label_dic=hypothesis_label_dic, random_seed=SEED_GLOBAL) # Label 0 means that the hypothesis is 'true', label 1 means that the hypothesis is 'not-true'

# Testing dataset
def format_rte_testset(df_test=None, hypo_label_dic=None):
  # Explode test dataset for N hypotheses
  hypothesis_lst = [value for key, value in hypo_label_dic.items()]

  # Label lists with 0 at alphabetical position of their true hypo, 1 for not-true hypos
  label_text_label_dic_explode = {}
  for key, value in hypo_label_dic.items():
    label_lst = [0 if value == hypo else 1 for hypo in hypothesis_lst]
    label_text_label_dic_explode[key] = label_lst

  df_test["label"] = df_test.label_text.map(label_text_label_dic_explode)
  df_test["hypothesis"] = [hypothesis_lst] * len(df_test)
  
  # Explode dataset to have K-1 additional rows with not_entail label and K-1 other hypotheses
  # After exploding, cannot sample anymore, because distorts the order to true label values, which needs to be preserved for evaluation code.
  df_test = df_test.explode(["hypothesis", "label"])
  
  df_test["label_rte_explicit"] = ["True" if label == 0 else "Not-True" for label in df_test["label"]]

  return df_test.copy(deep=True)

df_test_formatted = format_rte_testset(df_test=df_test, hypo_label_dic=hypothesis_label_dic)

# Load an RTE model

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model.to(device)

# Converting pandas dfs to Hugging Face dataset objects
import datasets

dataset = datasets.DatasetDict({
    "train": datasets.Dataset.from_pandas(df_train_formatted),
    "test": datasets.Dataset.from_pandas(df_test_formatted)
})

# Tokenize data
def tokenize_rte_format(examples):
  return tokenizer(examples["text"], examples["hypothesis"], truncation=True, max_length=512)

dataset["train"] = dataset["train"].map(tokenize_rte_format, batched=True)
dataset["test"] = dataset["test"].map(tokenize_rte_format, batched=True)

dataset = dataset.remove_columns(['column_name']) # Remove unnecessary columns.

# Set training hyperparameters
from transformers import TrainingArguments, Trainer, logging

training_directory = "name_your_directory"

fp16_bool = True if torch.cuda.is_available() else False
if "mdeberta" in model_name.lower(): fp16_bool = False
fp16_bool = False

train_args = TrainingArguments(
    output_dir=f'./results/{training_directory}',
    logging_dir=f'./logs/{training_directory}',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=80,
    gradient_accumulation_steps=2,
    num_train_epochs=8,
    warmup_ratio=0.06
    weight_decay=0.1,
    seed=SEED_GLOBAL,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=fp16_bool,
    fp16_full_eval=fp16_bool,
    evaluation_strategy="no",
    #eval_steps=10_000,
    save_strategy = "no",
    #save_steps=10_000,
    #save_total_limit=10,
    #logging_strategy="steps",
    report_to="all",
    #push_to_hub=False,
    #push_to_hub_model_id=f"{model_name}-finetuned-{task}",
)

# Helper function to clean memory and reduce risk of out-of-memory error
import gc
def clean_memory():
  #del(model)
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  gc.collect()

clean_memory()

# Function to compute metrics for RTE
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report

def compute_metrics_rte_binary(eval_pred, label_text_alphabetical=None):
    predictions, labels = eval_pred

    def chunks(lst, n):  # Yield successive n-sized chunks from lst. https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    softmax = torch.nn.Softmax(dim=1)
    prediction_chunks_lst = list(chunks(predictions, len(set(label_text_alphabetical))))
    hypo_position_highest_prob = []
    for i, chunk in enumerate(prediction_chunks_lst):
        hypo_position_highest_prob.append(np.argmax(np.array(chunk)[:, 0]))

    label_chunks_lst = list(chunks(labels, len(set(label_text_alphabetical))))
    label_position_gold = []
    for chunk in label_chunks_lst:
        label_position_gold.append(np.argmin(chunk))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='micro')
    acc_balanced = balanced_accuracy_score(label_position_gold, hypo_position_highest_prob)
    acc_not_balanced = accuracy_score(label_position_gold, hypo_position_highest_prob)
    metrics = {'accuracy_balanced': acc_balanced,
               'accuracy_not_b': acc_not_balanced,
               'precision_macro': precision_macro,
               'precision_micro': precision_micro,
               'recall_macro': recall_macro,
               'recall_micro': recall_micro,
               'f1_macro': f1_macro,
               'f1_micro': f1_micro,
               #'label_gold_raw': label_position_gold,
               #'label_predicted_raw': hypo_position_highest_prob
               }
    print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]})
    print("Detailed metrics: ", classification_report(label_position_gold, hypo_position_highest_prob, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2, output_dict=True,
                                zero_division='warn'), "\n")
    return metrics

# Create alphabetically ordered list of the original dataset classes/labels.
# This is necessary to be sure that the ordering of the test set labels and predictions is the same. Otherwise there is a risk that labels and predictions are in a different order and resulting metrics are wrong.
label_text_alphabetical = np.sort(df_train.label_text.unique())

# Training and evaluation
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=lambda eval_pred: compute_metrics_rte_binary(eval_pred, label_text_alphabetical=label_text_alphabetical)
)

trainer.train()

results = trainer.evaluate()
results = pd.DataFrame.from_dict(results, orient="index").reset_index()
results.rename(columns={"index": "metric", 0: "score"}, inplace=True)
print(results)