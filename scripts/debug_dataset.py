from datasets import load_dataset

def main():
    print("Loading dataset...")
    try:
        dataset = load_dataset("code_x_glue_ct_code_to_text", "python", split="train[:10]")
        print("Dataset loaded.")
        print(f"Length: {len(dataset)}")
        if len(dataset) > 0:
            item = dataset[0]
            print(f"Keys: {item.keys()}")
            print(f"Sample: {item}")
        else:
            print("Dataset is empty.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
