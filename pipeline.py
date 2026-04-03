import gzip
import csv
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s')

def batch_reader(filepath: str, batch_size: int = 1000) -> Iterator[List[Dict[str, str]]]:
    """
    Reads a gzipped CSV file and yields batches of rows as lists of dictionaries.
    This handles large files without loading everything into memory.
    """
    logging.info(f"Opening {filepath} for reading in batches of {batch_size}...")
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield the final partial batch if any
        if batch:
            yield batch

def mock_ai_sentiment_agent(review: Dict[str, str]) -> Dict[str, Any]:
    """
    A placeholder function simulating an AI Agent analyzing sentiment.
    In the real implementation, this would make an API call to an LLM.
    """
    # Simulate network/processing delay (e.g. inference time)
    # Using a very small sleep here just to simulate work without causing the script to hang forever on millions of rows.
    # time.sleep(0.001) 
    
    score = int(review.get('score', 0))
    # Naive mock logic purely based on star rating for demonstration
    if score >= 4:
        sentiment = 'positive'
    elif score <= 2:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
        
    return {
        'key': review.get('key'),
        'sentiment': sentiment,
        'confidence': 0.95, # Mock confidence
        'agent_version': '1.0-mock'
    }

def process_batch(batch_index: int, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    The Map task. Processes a batch of reviews.
    """
    logging.debug(f"Processing batch {batch_index} ({len(batch)} items) ...")
    start_time = time.time()
    
    # Process reviews in this batch
    results = [mock_ai_sentiment_agent(review) for review in batch]
    
    duration = time.time() - start_time
    logging.info(f"Batch {batch_index} processed in {duration:.2f} seconds.")
    return results

def run_pipeline(input_file: str, output_file: str, batch_size: int = 1000, max_workers: int = 4, max_batches: int = None):
    """
    The main orchestrator configuring the MapReduce pattern.
    """
    logging.info("Starting pipeline execution.")
    batches = batch_reader(input_file, batch_size)
    
    total_processed = 0
    total_batches = 0
    
    # Using ThreadPoolExecutor, well-suited for I/O bound tasks like making API requests to LLMs (AI Agents)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Dictionary to keep track of futures
        future_to_batch = {}
        
        # We start by buffering some work to the executor.
        for i, batch in enumerate(batches):
            future = executor.submit(process_batch, i, batch)
            future_to_batch[future] = i
            
            # Optional limit for testing/demonstration purposes
            if max_batches is not None and i + 1 >= max_batches:
                break
                
        # The Reduce step: Open output file in append/write mode
        with gzip.open(output_file, 'wt', encoding='utf-8') as out_f:
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    results = future.result()
                    
                    # Write results immediately to disk (Reduce and Sink)
                    for res in results:
                        out_f.write(json.dumps(res) + "\n")
                    
                    total_processed += len(results)
                    total_batches += 1
                    
                    if total_batches % 10 == 0:
                        logging.info(f"Progress: Processed {total_batches} batches ({total_processed} total reviews)")
                        
                except Exception as exc:
                    logging.error(f"Batch {batch_idx} generated an exception: {exc}")

    logging.info(f"Pipeline completed. Total records processed: {total_processed}. Results saved to {output_file}")


if __name__ == "__main__":
    INPUT_FILE = 'reviews_dataset.csv.gz'
    OUTPUT_FILE = 'sentiment_results.jsonl.gz'
    
    # Parameters for the pipeline
    # For a 3.3 million dataset, we might want larger batches or more workers depending on API rate limits.
    BATCH_SIZE = 5000
    WORKERS = 8
    
    # NOTE: Set max_batches to a small integer (e.g. 10) for local testing 
    # to process quickly instead of 3.3 million rows.
    # Set to None to process the entire file
    TEST_MODE_BATCHES = None 
    
    run_pipeline(
        input_file=INPUT_FILE, 
        output_file=OUTPUT_FILE, 
        batch_size=BATCH_SIZE, 
        max_workers=WORKERS,
        max_batches=TEST_MODE_BATCHES
    )
