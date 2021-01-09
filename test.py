from datasets import load_dataset, load_metric


metric = load_metric("f1", average="mcro")
print(metric)



