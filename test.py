from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, Trainer
import numpy as np

model = BertForSequenceClassification.from_pretrained('./sst2_bert_model', num_labels=2).to('cuda')
tokenizer = BertTokenizer.from_pretrained('./sst2_bert_model')

dataset = load_dataset('glue', 'sst2')
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

tokenized_test_dataset = dataset["validation"].map(tokenize_function, batched=True)
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# unique_labels = set(tokenized_test_dataset["label"])
# print(unique_labels)
# exit()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Set up dummy training arguments; required for Trainer but won't be used for evaluation
training_args = TrainingArguments(
    output_dir='./dummy_output',  # Dummy output directory
    do_train=False,
    do_eval=True,
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    eval_dataset=tokenized_test_dataset,
)

results = trainer.evaluate()
print(f"Accuracy on the test set: {results['eval_accuracy']:.4f}")
