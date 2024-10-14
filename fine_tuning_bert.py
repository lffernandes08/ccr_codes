import pandas as pd
import numpy as np

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

from transformers import set_seed
set_seed(42)

# Data download and preprocessing
df = pd.read_csv("your_dataset.csv")

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42) # Change the test set size

df_train = df_train[["label", "label_text", "text"]] # Suggested names for dataset columns
df_test = df_test[["label", "label_text", "text"]]

# Labels defined for vaccine-related headlines
label2id = {
    "global access to vaccine": 0,
    "science and technology": 1,
    "public health policies": 2,
    "vaccination rollout/campaign": 3,
    "vaccine hesitancy and mis-/disinformation": 4,
    "public endorsement to vaccine": 5,
    "institutional affairs": 6,
    "problems in vaccination": 7,
    "public perception of vaccine": 8,
    "economic consequences": 9
}

id2label = {v: k for k, v in label2id.items()}
id2label

# Load a Transformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch

model_name = "microsoft/deberta-v3-base"
#model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=512)
config = AutoConfig.from_pretrained(model_name, label2id=label2id, id2label=id2label, num_labels=len(label2id));
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True);

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model.to(device);

# Converting pandas dfs to Hugging Face dataset objects
import datasets

dataset = datasets.DatasetDict({
    "train": datasets.Dataset.from_pandas(df_train),
    "test": datasets.Dataset.from_pandas(df_test)
})

# Tokenize data
def tokenize(examples):
  return tokenizer(examples["text"], truncation=True, max_length=512)

dataset["train"] = dataset["train"].map(tokenize, batched=True)
dataset["test"] = dataset["test"].map(tokenize, batched=True)

dataset = dataset.remove_columns(['column_name']) # Remove unnecessary columns.

# Set training hyperparameters
training_directory = "name_your_directory"

fp16_bool = True if torch.cuda.is_available() else False
if "mdeberta" in model_name.lower(): fp16_bool = False
fp16_bool = False

from transformers import TrainingArguments, Trainer, logging

LEARNING_RATE = 2e-5
EPOCHS = 8

train_args = TrainingArguments(
    output_dir=f'./results/{training_directory}',
    logging_dir=f'./logs/{training_directory}',
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=80,
    gradient_accumulation_steps=2,
    warmup_ratio=0.06,
    weight_decay=0.1,
    seed=SEED_GLOBAL,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=fp16_bool,
    fp16_full_eval=fp16_bool,
    evaluation_strategy="epoch",
    #eval_steps=10_000,
    save_strategy = "epoch",
    #save_steps=10_000,
    #save_total_limit=10,
    #logging_strategy="steps",
    report_to="all",
    #push_to_hub=False,
    #push_to_hub_model_id=f"{model_name}-finetuned-{task}",
)

# Calculating metrics
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report
import warnings

def compute_metrics_standard(eval_pred):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        labels = eval_pred.label_ids
        pred_logits = eval_pred.predictions
        preds_max = np.argmax(pred_logits, axis=1)

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds_max, average='macro')
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds_max, average='micro')
        acc_balanced = balanced_accuracy_score(labels, preds_max)
        acc_not_balanced = accuracy_score(labels, preds_max)

        metrics = {
            'accuracy': acc_not_balanced,
            'f1_macro': f1_macro,
            'accuracy_balanced': acc_balanced,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
        }

        return metrics

# Training and evaluation
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics_standard
)

trainer.train()

results = trainer.evaluate()
print(results)
