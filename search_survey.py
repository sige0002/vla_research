#!/usr/bin/env python3
"""
Interdisciplinary Literature Survey for VLA Research Ideas
Searches arxiv and semantic scholar for relevant papers across multiple domains.
"""

import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
import time
import sys

def search_arxiv(query, max_results=5):
    """Search arxiv API for papers."""
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    url = base_url + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read().decode("utf-8")
        
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        
        results = []
        for entry in entries:
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")[:300]
            published = entry.find("atom:published", ns).text[:10]
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            link = entry.find("atom:id", ns).text
            results.append({
                "title": title,
                "authors": author_str,
                "published": published,
                "summary": summary,
                "link": link
            })
        return results
    except Exception as e:
        return [{"error": str(e), "query": query}]

def run_searches(category_name, queries):
    print(f"\n{'='*80}")
    print(f"## {category_name}")
    print(f"{'='*80}")
    
    all_results = {}
    for q in queries:
        print(f"\n--- Query: {q} ---")
        results = search_arxiv(q, max_results=3)
        time.sleep(1.5)  # rate limit
        all_results[q] = results
        for r in results:
            if "error" in r:
                print(f"  [ERROR] {r['error']}")
            else:
                print(f"  [{r['published']}] {r['title']}")
                print(f"    Authors: {r['authors']}")
                print(f"    Summary: {r['summary'][:200]}...")
                print(f"    Link: {r['link']}")
    return all_results

# Part 1: LLM/Transformer Memory Research
part1_queries = [
    "episodic memory transformer neural network",
    "working memory large language model",
    "memory consolidation neural network sleep replay",
    "retrieval augmented generation memory robot",
    "associative memory transformer attention",
    "hippocampal memory replay artificial intelligence",
    "memory-augmented neural network continual learning",
]

# Part 2: Human Task Cognition
part2_queries = [
    "human task decomposition cognitive science",
    "cognitive architecture task planning robot",
    "affordance perception cognitive development robot",
    "schema theory motor learning robot",
    "predictive coding motor control robot",
    "embodied cognition grounded language robot",
    "developmental learning robot manipulation",
    "infant object manipulation cognitive development",
    "mirror neuron system robot imitation learning",
]

# Part 3: Neuroscience-Inspired Robot Control
part3_queries = [
    "cerebellum model robot control forward model",
    "basal ganglia reinforcement learning robot",
    "prefrontal cortex working memory robot planning",
    "predictive processing active inference robot",
    "active inference robot manipulation",
    "free energy principle robot control",
    "hierarchical predictive coding robot",
]

# Part 4: Novel AI Techniques
part4_queries = [
    "test time training adaptation robot",
    "in-context learning robot manipulation",
    "meta-learning robot few-shot manipulation",
    "curiosity driven exploration robot manipulation",
    "compositional generalization robot manipulation",
    "neuro-symbolic robot planning",
    "mixture of experts robot policy",
    "foundation model adaptation continual learning robot",
]

all_data = {}
all_data["Part1_Memory"] = run_searches("Part 1: LLM/Transformer Memory Research", part1_queries)
all_data["Part2_Cognition"] = run_searches("Part 2: Human Task Cognition & Development", part2_queries)
all_data["Part3_Neuroscience"] = run_searches("Part 3: Neuroscience-Inspired Robot Control", part3_queries)
all_data["Part4_Novel_AI"] = run_searches("Part 4: Novel AI Techniques for VLA", part4_queries)

# Save raw data
with open("survey_results.json", "w") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print("\n\nSurvey complete. Results saved to survey_results.json")
