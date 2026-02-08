import sys
sys.path.insert(0, '/mnt/project')

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
import json
import time
from datetime import datetime
import os


class GraphRAGIndexer:
    """
    Enhanced RAG Indexing mit LLM-Analyse
    """
    
    def __init__(
        self,
        llm_base_url: str = "XXX",
        llm_model: str = "gpt-oss:20b",
        persist_directory: str = "./chroma_db_graph_rag"
    ):
        self.persist_directory = persist_directory
        
        print(" Initialisiere Graph-RAG Indexer...")
        
       
        print(" Verbinde mit LLM...")
        self.llm = ChatOllama(
            model=llm_model,
            base_url=llm_base_url,
            timeout=120,
            temperature=0
        )
        
        
        print("🧬 Lade Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectordb = None
        
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=200,  
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )
        
        print(" Graph-RAG Indexer bereit!\n")
    
    def analyze_chunk_with_llm(self, chunk: Document) -> Dict:
        """
        Nutzt LLM um Chunk semantisch zu analysieren.
        
        Extrahiert:
        - Zusammenfassung (1-2 Sätze)
        - Keywords/Tags (max 10)
        - Entities (Tools, Technologien, Konzepte)
        - Kategorie (deployment/security/networking/etc)
        - Schwierigkeitsgrad
        - Ob Code enthalten ist
        """
        
        content = chunk.page_content[:2000]  
        
        prompt = f"""Analyze this technical documentation chunk and extract structured metadata.

DOCUMENTATION CHUNK:
{content}

Extract the following as JSON (be concise and precise):

{{
  "summary": "One sentence summary of what this chunk teaches",
  "keywords": ["keyword1", "keyword2", ...],  // Max 10, most relevant
  "entities": {{
    "tools": ["tool1", "tool2"],  // e.g. kubectl, docker, ollama
    "technologies": ["tech1"],    // e.g. kubernetes, gpu, yaml
    "concepts": ["concept1"]      // e.g. deployment, security, networking
  }},
  "category": "deployment|security|networking|storage|ml-ops|operations|fundamentals",
  "difficulty": "beginner|intermediate|advanced",
  "has_code": true|false,
  "code_languages": ["yaml", "bash", "python"],  // if has_code=true
  "main_topic": "Brief topic description"
}}

Return ONLY valid JSON, no markdown, no explanation.
"""
        
        try:
            response = self.llm.invoke(prompt)
            
           
            content = response.content.strip()
            
            
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:].strip()
            
            metadata = json.loads(content)
            
            
            if not isinstance(metadata.get('keywords'), list):
                metadata['keywords'] = []
            if not isinstance(metadata.get('entities'), dict):
                metadata['entities'] = {'tools': [], 'technologies': [], 'concepts': []}
            
            return metadata
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️  JSON Parse Error: {e}")
            print(f"   Raw response: {response.content[:200]}")
            return self._get_fallback_metadata()
        except Exception as e:
            print(f"   ⚠️  LLM Analysis failed: {e}")
            return self._get_fallback_metadata()
    
    def _get_fallback_metadata(self) -> Dict:
        """Fallback wenn LLM-Analyse fehlschlägt"""
        return {
            'summary': 'Technical documentation chunk',
            'keywords': [],
            'entities': {'tools': [], 'technologies': [], 'concepts': []},
            'category': 'fundamentals',
            'difficulty': 'intermediate',
            'has_code': False,
            'code_languages': [],
            'main_topic': 'General'
        }
    
    def enhance_chunk(self, chunk: Document) -> Document:
        """
        Enhanced ein Chunk mit LLM-generierten Metadaten.
        
        Returns:
            Document mit erweiterten Metadaten
        """
        
        print(f"   🔍 Analyzing chunk (len={len(chunk.page_content)})...")
        
        
        llm_metadata = self.analyze_chunk_with_llm(chunk)
        
        def list_to_string(lst):
            if isinstance(lst, list):
                return ", ".join(str(item) for item in lst)
            return str(lst) if lst else ""
        
        
        enhanced_metadata = {
            **chunk.metadata, 
            'llm_summary': llm_metadata['summary'],
            'llm_keywords': list_to_string(llm_metadata['keywords']), 
            'llm_entities_tools': list_to_string(llm_metadata['entities'].get('tools', [])),  
            'llm_entities_technologies': list_to_string(llm_metadata['entities'].get('technologies', [])),  
            'llm_entities_concepts': list_to_string(llm_metadata['entities'].get('concepts', [])),  
            'llm_category': llm_metadata['category'],
            'llm_difficulty': llm_metadata['difficulty'],
            'llm_has_code': llm_metadata['has_code'],
            'llm_code_languages': list_to_string(llm_metadata.get('code_languages', [])), 
            'llm_main_topic': llm_metadata['main_topic'],
            'enhanced_at': datetime.now().isoformat()
        }
        
        
        enhanced_content = f"""[SUMMARY] {llm_metadata['summary']}

[KEYWORDS] {', '.join(llm_metadata['keywords'])}

[ORIGINAL CONTENT]
{chunk.page_content}"""
        
        enhanced_chunk = Document(
            page_content=enhanced_content,
            metadata=enhanced_metadata
        )
        
        print(f"      ✓ Category: {llm_metadata['category']}")
        keywords_str = list_to_string(llm_metadata['keywords'])
        
        if keywords_str:
            
            keywords_list = [k.strip() for k in keywords_str.split(',')][:5]
            print(f"      ✓ Keywords: {', '.join(keywords_list)}")
        else:
            print(f"      ✓ Keywords: (none)")
        
        return enhanced_chunk
    
    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 5,
        show_progress: bool = True
    ) -> int:
        """
        Indexiert Dokumente mit LLM-Enhancement.
        
        Args:
            documents: Liste von Source-Documents
            batch_size: Wie viele Chunks pro LLM-Batch
            show_progress: Fortschritt anzeigen
        
        Returns:
            Anzahl indexierter Chunks
        """
        
        print("="*80)
        print(" GRAPH-RAG INDEXING PIPELINE")
        print("="*80 + "\n")
        
       
        print("  STEP 1: Chunking documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"   ✓ Created {len(chunks)} chunks\n")
        
        
        print(" STEP 2: LLM Enhancement (this will take time)...")
        print(f"   Processing {len(chunks)} chunks with LLM analysis...")
        print(f"   Estimated time: ~{len(chunks) * 3} seconds\n")
        
        enhanced_chunks = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks, 1):
            if show_progress and i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(chunks) - i) / rate if rate > 0 else 0
                print(f"   Progress: {i}/{len(chunks)} ({i/len(chunks)*100:.1f}%) | "
                      f"Rate: {rate:.1f} chunks/sec | "
                      f"ETA: {remaining:.0f}s")
            
            try:
                enhanced = self.enhance_chunk(chunk)
                enhanced_chunks.append(enhanced)
            except Exception as e:
                print(f"     Error enhancing chunk {i}: {e}")
                
                enhanced_chunks.append(chunk)
            
           
            time.sleep(0.5)
        
        duration = time.time() - start_time
        print(f"\n    Enhanced {len(enhanced_chunks)} chunks in {duration:.1f}s\n")
        
    
        print(" STEP 3: Creating Vector Database...")
        
        self.vectordb = Chroma.from_documents(
            documents=enhanced_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"    Vector DB created at: {os.path.abspath(self.persist_directory)}\n")
        
        
        print("="*80)
        print(" INDEXING STATISTICS")
        print("="*80)
        print(f"Total Chunks: {len(enhanced_chunks)}")
        print(f"Processing Time: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"Average per Chunk: {duration/len(chunks):.2f}s")
        
        
        categories = [c.metadata.get('llm_category', 'unknown') for c in enhanced_chunks]
        from collections import Counter
        cat_counts = Counter(categories)
        
        print(f"\nCategory Distribution:")
        for cat, count in sorted(cat_counts.items()):
            print(f"  - {cat}: {count}")
        
        
        difficulties = [c.metadata.get('llm_difficulty', 'unknown') for c in enhanced_chunks]
        diff_counts = Counter(difficulties)
        
        print(f"\nDifficulty Distribution:")
        for diff, count in sorted(diff_counts.items()):
            print(f"  - {diff}: {count}")
        
        print("\n" + "="*80)
        print(" GRAPH-RAG INDEXING COMPLETE!")
        print("="*80 + "\n")
        
        return len(enhanced_chunks)
    
    def load_existing(self) -> bool:
        """Lädt existierende Graph-RAG DB"""
        if os.path.exists(self.persist_directory):
            print(f" Loading existing Graph-RAG DB from {self.persist_directory}...")
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(" Graph-RAG DB loaded\n")
            return True
        else:
            print(f"  No Graph-RAG DB found at {self.persist_directory}\n")
            return False
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        must_have_code: Optional[bool] = None
    ) -> List[Document]:
        """
        Hybrid Search: Vector Similarity + Metadata Filtering
        
        Args:
            query: Suchanfrage
            k: Anzahl Ergebnisse
            category: Filter nach Kategorie
            difficulty: Filter nach Schwierigkeitsgrad
            must_have_code: Nur Chunks mit Code
        
        Returns:
            Liste relevanter Dokumente
        """
        
        if self.vectordb is None:
            raise ValueError("Vector DB not loaded!")
        
      
        filter_conditions = []
        
        if category:
            filter_conditions.append({'llm_category': category})
        if difficulty:
            filter_conditions.append({'llm_difficulty': difficulty})
        if must_have_code is not None:
            filter_conditions.append({'llm_has_code': must_have_code})
        
        
        if len(filter_conditions) == 0:
            
            filter_dict = None
        elif len(filter_conditions) == 1:
            
            filter_dict = filter_conditions[0]
        else:
            
            filter_dict = {'$and': filter_conditions}
        
        
        if filter_dict:
            results = self.vectordb.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vectordb.similarity_search(query, k=k)
        
        return results
    
    def smart_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Intelligente Suche die automatisch Query analysiert.
        
        Erkennt z.B.:
        - "beginner tutorial for..." -> filtered by difficulty
        - "security best practices" -> filtered by category
        - "yaml example for..." -> must_have_code=True
        """
        
        query_lower = query.lower()
        
        
        category = None
        difficulty = None
        must_have_code = None
        
        
        if any(word in query_lower for word in ['beginner', 'basic', 'introduction', 'getting started']):
            difficulty = 'beginner'
        elif any(word in query_lower for word in ['advanced', 'complex', 'production']):
            difficulty = 'advanced'
        
        
        if 'security' in query_lower:
            category = 'security'
        elif any(word in query_lower for word in ['deploy', 'deployment']):
            category = 'deployment'
        elif 'network' in query_lower:
            category = 'networking'
        elif any(word in query_lower for word in ['gpu', 'ml', 'ollama', 'model']):
            category = 'ml-ops'
        
        
        if any(word in query_lower for word in ['yaml', 'code', 'example', 'configuration', 'script']):
            must_have_code = True
        
        print(f" Smart Search Filters:")
        print(f"   Category: {category or 'any'}")
        print(f"   Difficulty: {difficulty or 'any'}")
        print(f"   Must have code: {must_have_code or 'no preference'}")
        print()
        
        return self.hybrid_search(
            query=query,
            k=k,
            category=category,
            difficulty=difficulty,
            must_have_code=must_have_code
        )



def rebuild_graph_rag_from_existing():
    """
    Rebuilt Graph-RAG Index aus existierenden Source-Dokumenten.
    """
    
    from rag_system_crawl import EnhancedRAGSystem
    
    print("\n" + "="*80)
    print(" REBUILDING GRAPH-RAG INDEX FROM SOURCES")
    print("="*80 + "\n")
    
    
    print(" Loading source documents...")
    rag_original = EnhancedRAGSystem()
    
    
    md_docs = rag_original.load_markdown_documents()
    print(f"    Loaded {len(md_docs)} markdown files")
    
    
    pdf_docs = rag_original.load_pdf_documents()
    print(f"    Loaded {len(pdf_docs)} PDF pages")
    
    all_docs = md_docs + pdf_docs
    
    if not all_docs:
        print("\n No documents found! Check your knowledge_base/ directory.")
        return
    
    print(f"\n Total source documents: {len(all_docs)}\n")
    
    
    graph_rag = GraphRAGIndexer(
        persist_directory="./chroma_db_graph_rag"
    )
    
   
    num_chunks = graph_rag.index_documents(
        documents=all_docs,
        batch_size=5,
        show_progress=True
    )
    
    print(f"\n Graph-RAG Index created with {num_chunks} enhanced chunks!")
    
    return graph_rag


if __name__ == "__main__":
    
    
    graph_rag = rebuild_graph_rag_from_existing()
    
    if graph_rag and graph_rag.vectordb:
        
        print("\n" + "="*80)
        print(" TEST SEARCHES")
        print("="*80 + "\n")
        
        test_queries = [
            "How to deploy Ollama with GPU?",
            "Security best practices for Kubernetes",
            "YAML configuration for Flask app",
            "Beginner tutorial for deployments"
        ]
        
        for query in test_queries:
            print(f" Query: '{query}'")
            print("-"*80)
            
            results = graph_rag.smart_search(query, k=3)
            
            if not results:
                print("    No results found\n")
                continue
            
            for i, doc in enumerate(results, 1):
                keywords_str = doc.metadata.get('llm_keywords', '')
                
                if keywords_str:
                    keywords_list = [k.strip() for k in keywords_str.split(',')][:5]
                else:
                    keywords_list = []
                
                print(f"\n[{i}] Category: {doc.metadata.get('llm_category', 'N/A')} | "
                      f"Difficulty: {doc.metadata.get('llm_difficulty', 'N/A')}")
                print(f"    Keywords: {', '.join(keywords_list) if keywords_list else '(none)'}")
                print(f"    Summary: {doc.metadata.get('llm_summary', 'N/A')}")
                
                
                original_content = doc.page_content.split('[ORIGINAL CONTENT]')[-1].strip()
                print(f"    Content: {original_content[:150]}...")
            
            print("\n")
