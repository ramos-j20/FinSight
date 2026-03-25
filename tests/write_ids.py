import json
from backend.agent.retriever import FinSightRetriever

def main():
    print("Initializing retriever...")
    retriever = FinSightRetriever()
    
    in_path = "/app/tests/fixtures/eval_dataset.jsonl"
    out_path = "/app/tests/fixtures/new_eval_dataset.jsonl"
    
    updated_lines = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            
            print(f"Querying: {data['query']}")
            chunks = retriever.retrieve(
                query=data["query"], 
                top_k=2, 
                ticker_filter=data.get("ticker_filter")
            )
            
            if chunks:
                actual_ids = [c.chunk_id for c in chunks]
                data["expected_chunk_ids"] = actual_ids
                print(f"  -> Found IDs: {actual_ids}")
            else:
                print("  -> ERROR: No chunks found!")
                
            updated_lines.append(json.dumps(data))
            
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")
    print("Done writing to new_eval_dataset.jsonl")

if __name__ == "__main__":
    main()
