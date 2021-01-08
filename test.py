from datasets import load_dataset, load_metric


metric = load_metric("glue", "mrpc")
print(metric)



