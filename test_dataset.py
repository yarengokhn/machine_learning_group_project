from datasets import load_dataset

# Python veri setini indirelim
dataset = load_dataset("Nan-Do/code-search-net-python")

# Veri setinin yapısına bakalım
print(dataset)

# Örnek bir veriyi inceleyelim
sample = dataset['train'][0]
print("--- KOD ---")
print(sample['code'])
print("\n--- ÖZET (DOCSTRING) ---")
print(sample['summary'])