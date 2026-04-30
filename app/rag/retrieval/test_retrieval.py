#!/usr/bin/env python3
"""
Demo script để kiểm tra hệ thống retrieval.

Cách chạy:
    python app/rag/test_retrieval.py
"""

import sys
import os
from app.rag.retrieval import get_retriever, search, search_by_category


def main():
    print("=" * 70)
    print("🧠 DEMO: Language Learning Agent - Retrieval System")
    print("=" * 70)
    
    # Khởi tạo retriever
    print("\n📚 Initializing retriever...")
    retriever = get_retriever(use_ivf=True)
    
    # Hiển thị thống kê
    stats = retriever.get_stats()
    print("\n📊 Index Statistics:")
    print(f"   • Total vectors: {stats['n_vectors']:,}")
    print(f"   • Embedding dimension: {stats['dimension']}")
    print(f"   • Index type: {stats['index_type']}")
    print(f"   • Is optimized (IVF): {stats['is_optimized']}")
    if stats['nprobe']:
        print(f"   • nprobe (clusters to check): {stats['nprobe']}")
    
    # Kiểm tra xem có dữ liệu không
    if stats['n_vectors'] == 0:
        print("\n⚠️  No vectors in index! Run ingestion first:")
        print("   python -m app.rag.ingest")
        return
    
    # Test retrieval
    print("\n" + "=" * 70)
    print("🔍 Testing Retrieval")
    print("=" * 70)
    
    test_queries = [
        "Cách nói xin lỗi",
        "Động từ trong tiếng Bồ Đào Nha",
        "Số từ 1 đến 10",
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        print("-" * 70)
        
        results = search(query, k=3)
        
        if not results:
            print("   ❌ No results found")
            continue
        
        for i, result in enumerate(results, 1):
            print(f"\n   {i}. Score: {result['score']:.4f}")
            text_preview = result['text'][:80].replace('\n', ' ')
            print(f"      Text: {text_preview}...")
            print(f"      ID: {result['id'][:16]}...")
            metadata = result['metadata']
            print(f"      Metadata: {metadata}")
    
    # Test filtering
    print("\n" + "=" * 70)
    print("🏷️  Testing Metadata Filtering")
    print("=" * 70)
    
    print("\n❓ Query: 'Tenses' with type='passage'")
    results = search("Tenses", k=3, filters={"type": "passage"})
    print(f"   Found {len(results)} results")
    for i, r in enumerate(results[:2], 1):
        print(f"   {i}. {r['text'][:60]}... (type: {r['metadata'].get('type')})")
    
    # Test batch retrieval
    print("\n" + "=" * 70)
    print("📦 Testing Batch Retrieval")
    print("=" * 70)
    
    batch_queries = ["verb", "subject", "object"]
    print(f"\nQueries: {batch_queries}")
    batch_results = retriever.batch_retrieve(batch_queries, k=2)
    
    for query, results in batch_results.items():
        print(f"\n   '{query}': {len(results)} results")
        if results:
            print(f"      → {results[0]['text'][:60]}...")
    
    # Benchmark
    print("\n" + "=" * 70)
    print("⏱️  Benchmarking Search Speed")
    print("=" * 70)
    
    benchmark_query = "learning Portuguese vocabulary"
    print(f"\nBenchmarking with: '{benchmark_query}'")
    print("Running 5 iterations...\n")
    
    bench = retriever.benchmark_search(benchmark_query, iterations=5)
    print(f"   Min time:  {bench['min']*1000:.2f}ms")
    print(f"   Avg time:  {bench['avg']*1000:.2f}ms")
    print(f"   Max time:  {bench['max']*1000:.2f}ms")
    print(f"   Total:     {bench['total']:.3f}s")
    
    if bench['avg'] < 0.1:
        print(f"\n   ✅ Search speed is excellent (< 0.1s)")
    elif bench['avg'] < 0.2:
        print(f"\n   ✓ Search speed is good (< 0.2s)")
    else:
        print(f"\n   ⚠️  Search is slow. Consider optimizing with more IVF clusters")
    
    # Integration test
    print("\n" + "=" * 70)
    print("✨ Integration Test")
    print("=" * 70)
    
    print("\nSimulating RAG pipeline:")
    user_query = "Como se diz hello em português?"
    print(f"User: {user_query}")
    
    retrieved = search(user_query, k=3)
    
    if retrieved:
        print(f"\n🔍 Retrieved {len(retrieved)} relevant chunks:")
        
        context = "\n---\n".join([
            f"[Score: {r['score']:.3f}] {r['text'][:100]}"
            for r in retrieved
        ])
        
        print(f"\nContext for LLM:\n{context}")
        
        print("\n\n💬 LLM would now use this context to answer:")
        print(f"   'Olá' means hello/hi in Portuguese")
    else:
        print("❌ Could not retrieve relevant chunks")
    
    print("\n" + "=" * 70)
    print("✅ Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
