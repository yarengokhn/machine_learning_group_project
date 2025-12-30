import pandas as pd
import gzip
import os

def prepare_data():
    base_path = "data/python/original data"
    output_path = "data/dataset.csv"
    
    print("Reading declarations...")
    with open(os.path.join(base_path, "updated_parellel_decl"), 'r', encoding='utf-8') as f:
        decls = [line.strip() for line in f]
        
    print("Reading summaries...")
    with open(os.path.join(base_path, "updated_parellel_desc"), 'r', encoding='utf-8') as f:
        summaries = [line.strip() for line in f]
        
    print("Reading bodies...")
    with gzip.open(os.path.join(base_path, "updated_parellel_bodies.gz"), 'rt', encoding='utf-8') as f:
        bodies = [line.strip() for line in f]
        
    print(f"Loaded {len(decls)} decls, {len(summaries)} summaries, {len(bodies)} bodies.")
    
    if not (len(decls) == len(summaries) == len(bodies)):
        print("Error: File lengths do not match!")
        return

    data = []
    print("Processing data...")
    for decl, summary, body in zip(decls, summaries, bodies):
        # Reconstruct code: signature + newline + body
        # Replace special tokens in body
        # DCNL -> \n
        # DCSP -> indentation (4 spaces)
        
        processed_body = body.replace('DCNL', '\n').replace('DCSP', '    ')
        full_code = f"{decl}\n{processed_body}"
        
        # Clean up summary (it also has DCNL sometimes based on view_file output)
        processed_summary = summary.replace('DCNL', ' ')
        
        data.append({
            "code": full_code,
            "summary": processed_summary
        })
        
    df = pd.DataFrame(data)
    
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    prepare_data()
