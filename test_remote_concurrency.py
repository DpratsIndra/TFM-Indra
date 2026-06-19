import os
import time
import requests
import concurrent.futures
from dotenv import load_dotenv

def test_vllm(req_id, url, model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Hi, just testing concurrency. Request ID {req_id}. Please reply with a short message."}],
        "max_tokens": 50
    }
    t0 = time.time()
    try:
        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        return req_id, "SUCCESS", time.time() - t0
    except Exception as e:
        return req_id, f"ERROR: {e}", time.time() - t0

def test_reranker(req_id, url, model):
    payload = {
        "model": model,
        "query": f"Cybersecurity test query {req_id}",
        "documents": [
            f"This is a relevant document for test {req_id}", 
            f"This is an irrelevant random sentence for test {req_id}"
        ]
    }
    t0 = time.time()
    try:
        res = requests.post(url, json=payload, headers={"Authorization": "Bearer EMPTY"}, timeout=60)
        res.raise_for_status()
        return req_id, "SUCCESS", time.time() - t0
    except Exception as e:
        return req_id, f"ERROR: {e}", time.time() - t0

def test_embeddings(req_id, url, model):
    payload = {
        "model": model,
        "input": [f"This is a test sentence for embeddings number {req_id}"]
    }
    t0 = time.time()
    try:
        res = requests.post(url, json=payload, headers={"Authorization": "Bearer EMPTY"}, timeout=60)
        res.raise_for_status()
        return req_id, "SUCCESS", time.time() - t0
    except Exception as e:
        return req_id, f"ERROR: {e}", time.time() - t0

def main():
    load_dotenv(override=True)
    
    # Configure URLs based on .env
    use_gemma = os.getenv("USE_GEMMA4", "False").lower() in ("true", "1", "yes")
    if use_gemma:
        vllm_url = os.getenv("VLLM_BASE_URL_GEMMA", "http://10.0.152.198:8003/v1").rstrip("/") + "/chat/completions"
        vllm_model = os.getenv("VLLM_MODEL_NAME_GEMMA", "gemma4")
    else:
        vllm_url = os.getenv("VLLM_BASE_URL", "http://10.0.152.198:8001/v1").rstrip("/") + "/chat/completions"
        vllm_model = os.getenv("VLLM_MODEL_NAME", "gpt-oss-20b")
    
    reranker_url = os.getenv("RERANKER_URL", "http://10.0.152.198:8005/v1/rerank")
    reranker_model = os.getenv("RERANKER_MODEL_NAME", "jina-reranker-v2-base-multilingual")

    embeddings_url = os.getenv("EMBEDDINGS_BASE_URL", "http://10.0.152.198:8002/v1").rstrip("/") + "/embeddings"
    embeddings_model = os.getenv("EMBEDDINGS_MODEL_NAME", "BAAI/bge-m3")
    
    CONCURRENT_REQUESTS = 15

    print("="*60)
    print(f"CONCURRENCY STRESS TEST: {CONCURRENT_REQUESTS} REQUESTS")
    print("="*60)
    
    # 1. Test LLM Endpoint
    print(f"\n[1] Testing vLLM Endpoint: {vllm_url} (Model: {vllm_model})")
    start_llm = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(test_vllm, i, vllm_url, vllm_model) for i in range(CONCURRENT_REQUESTS)]
        for f in concurrent.futures.as_completed(futures):
            req_id, status, duration = f.result()
            if status == "SUCCESS":
                print(f"  [+] Req {req_id:02d} | Status: {status} | Time: {duration:.2f}s")
            else:
                print(f"  [-] Req {req_id:02d} | Status: {status} | Time: {duration:.2f}s")
                
    print(f"-> LLM Test Finished. Total time: {time.time() - start_llm:.2f}s")

    # 2. Test Reranker Endpoint
    print(f"\n[2] Testing TEI Reranker Endpoint: {reranker_url} (Model: {reranker_model})")
    start_rerank = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(test_reranker, i, reranker_url, reranker_model) for i in range(CONCURRENT_REQUESTS)]
        for f in concurrent.futures.as_completed(futures):
            req_id, status, duration = f.result()
            if status == "SUCCESS":
                print(f"  [+] Req {req_id:02d} | Status: {status} | Time: {duration:.2f}s")
            else:
                print(f"  [-] Req {req_id:02d} | Status: {status} | Time: {duration:.2f}s")
                
    print(f"-> Reranker Test Finished. Total time: {time.time() - start_rerank:.2f}s")
    
    # 3. Test Embeddings Endpoint
    print(f"\n[3] Testing TEI Embeddings Endpoint: {embeddings_url} (Model: {embeddings_model})")
    start_embed = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(test_embeddings, i, embeddings_url, embeddings_model) for i in range(CONCURRENT_REQUESTS)]
        for f in concurrent.futures.as_completed(futures):
            req_id, status, duration = f.result()
            if status == "SUCCESS":
                print(f"  [+] Req {req_id:02d} | Status: {status} | Time: {duration:.2f}s")
            else:
                print(f"  [-] Req {req_id:02d} | Status: {status} | Time: {duration:.2f}s")
                
    print(f"-> Embeddings Test Finished. Total time: {time.time() - start_embed:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
