from langchain_core.tools import tool
import subprocess
import json
import yaml
import sys
import os
import re

search_metadata = {
    'num_query_variants': 0,
    'query_variants': [],
    'total_chunks_evaluated': 0,
    'unique_chunks_found': 0,
    'final_chunks_returned': 0,
    'top_scores': [],
    'search_duration_ms': 0
}

def reset_search_metadata():
    """Setzt Search-Metadaten zurück."""
    global search_metadata
    search_metadata = {
        'num_query_variants': 0,
        'query_variants': [],
        'total_chunks_evaluated': 0,
        'unique_chunks_found': 0,
        'final_chunks_returned': 0,
        'top_scores': [],
        'search_duration_ms': 0
    }

def get_search_metadata() -> dict:
    """Gibt aktuelle Search-Metadaten zurück."""
    return search_metadata.copy()


sys.path.insert(0, '/mnt/user-data/outputs')

try:
    from enhanced_rag_system_v2 import EnhancedRAGSystemV2 as RAGSystem
    print("🔥 Initialisiere Graph-RAG System...")
    rag = RAGSystem(persist_directory="./chroma_db_graph_rag")
    
    if rag.load_existing():
        print("✅ Graph-RAG System bereit!")
        
        stats = rag.get_source_statistics()
        print(f"   📊 {stats['total_chunks']} Chunks")
        if 'by_category' in stats:
            print(f"   📂 Kategorien: {len(stats['by_category'])}")
    else:
        print("⚠️  Graph-RAG DB nicht gefunden!")
        print("   Führe zuerst aus: python graph_rag_indexer.py")
        from rag_system_crawl import EnhancedRAGSystem as OldRAG
        rag = OldRAG()
        rag.load_existing()
        print("   ℹ️  Nutze Fallback RAG System")
except ImportError:
    print("⚠️  Graph-RAG nicht verfügbar, nutze altes System")
    from rag_system_crawl import EnhancedRAGSystem as RAGSystem
    rag = RAGSystem()
    rag.load_existing()
    print("✅ Standard RAG System bereit")

print()

from tutorial_navigator import TutorialNavigator
print("🔧 Initialisiere Tutorial Navigator...")
tutorial_nav = TutorialNavigator()
print("✅ Tutorial Navigator bereit!\n")


def extract_commands_from_text(text: str) -> list:
    """
    Extrahiert ausführbare Befehle aus Text.
    
    Erkennt:
    - Code-Blöcke (```bash, ```yaml)
    - Nummerierte Zeilen aus PDFs
    - Inline-Befehle
    """
    commands = []
    
    
    code_blocks = re.findall(r'```(?:bash|shell|yaml|python)?\n?(.*?)```', text, re.DOTALL)
    for block in code_blocks:
        for line in block.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                commands.append(line)
    
    
    lines = text.split('\n')
    for line in lines:
        match = re.match(r'^\s*[0-9]+\s*([a-z./~][^\n]{5,})$', line.strip(), re.IGNORECASE)
        if match:
            cmd = match.group(1).strip()
            
            indicators = ['pip', 'git', 'cd', 'copier', 'oc', 'kubectl', 'docker', 
                          'podman', 'curl', 'wget', 'unzip', 'tar', 'echo', 'cat',
                          'mkdir', 'rm', 'cp', 'mv', 'source', 'export', 'rscript',
                          './', 'python', 'npm', 'yarn', 'from ', 'copy ', 'run ']
            if any(ind in cmd.lower() for ind in indicators):
                commands.append(cmd)
    
    return list(dict.fromkeys(commands))  


def is_table_of_contents(text: str) -> bool:
    """Erkennt ob Text ein Inhaltsverzeichnis ist."""
    toc_words = ['Überblick', 'Vorbedingungen', 'Ablauf', 'Inhaltsverzeichnis']
    count = sum(1 for w in toc_words if w in text)
    
    lines = text.strip().split('\n')
    short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 40)
    
    return count >= 2 and short_lines > len(lines) * 0.4


def score_chunk(text: str, query: str, metadata: dict = None) -> float:
    """
    Bewertet einen Chunk nach RELEVANZ zur Query.
    
    Args:
        text: Chunk-Inhalt
        query: Original-Query
        metadata: Graph-RAG Metadaten (optional)
    
    Returns:
        Score (höher = relevanter)
    """
    score = 50.0
    text_lower = text.lower()
    query_lower = query.lower()
    
    query_words = [w for w in query_lower.split() 
                   if w not in ['der', 'die', 'das', 'ein', 'eine', 'mit', 
                                'und', 'oder', 'für', 'via', 'einer', 'automatisches',
                                'deployment', 'wie', 'kann', 'ich']]
    
    
    if metadata:
        
        llm_category = metadata.get('llm_category', '')
        if llm_category in query_lower:
            score += 40
        
        llm_difficulty = metadata.get('llm_difficulty', '')
        if 'beginner' in query_lower and llm_difficulty == 'beginner':
            score += 30
        
        
        llm_has_code = metadata.get('llm_has_code', False)
        if llm_has_code and any(word in query_lower for word in ['yaml', 'code', 'beispiel', 'example', 'config']):
            score += 35
        
        
        llm_keywords_str = metadata.get('llm_keywords', '')
        if llm_keywords_str:
            llm_keywords = [k.strip().lower() for k in llm_keywords_str.split(',')]
            matching_keywords = sum(1 for kw in llm_keywords if any(qw in kw for qw in query_words))
            score += matching_keywords * 15
    
    
    for word in query_words:
        if word in text_lower:
            score += 30
    
    commands = extract_commands_from_text(text)
    relevant_commands = 0
    for cmd in commands:
        cmd_lower = cmd.lower()
        if any(word in cmd_lower for word in query_words):
            relevant_commands += 1
            score += 40
        else:
            if any(df in cmd_lower for df in ['from ', 'copy ', 'run ', 'env ', 'workdir']):
                score += 5
            else:
                score += 15
    
    for pattern in ['Geben Sie', 'Führen Sie', 'Installieren Sie', 
                    'Klicken Sie', 'Erstellen Sie', 'Wechseln Sie',
                    'Step 1', 'Step 2', 'Schritt 1', 'Schritt 2']:
        if pattern in text:
            score += 10
    
    if 'copier' in query_lower:
        if 'pip install copier' in text_lower:
            score += 50
        if 'copier copy' in text_lower:
            score += 50
        if 'git clone' in text_lower:
            score += 30
    
    if 'shiny' in query_lower:
        if 'shiny' in text_lower and 'copier' in text_lower:
            score += 40
    
    if 'ollama' in query_lower:
        if 'ollama' in text_lower and 'gpu' in text_lower:
            score += 40
    
    
    if is_table_of_contents(text):
        score -= 100
    
    
    if any(x in text_lower for x in ['impressum', 'datenschutz', 'barrierefreiheit']):
        score -= 80
    
    
    if len(text) < 200:
        score -= 30
    
    
    if 'dockerfile' in text_lower and relevant_commands == 0:
        score -= 20
    
    return score


def generate_query_variants(query: str) -> list:
    """
    Generiert Query-Varianten für Multi-Query-Suche.
    
    Strategie:
    - Original Query
    - Variations mit Kontext-Keywords
    - Spezifische Befehls-Suchen
    """
    variants = [query]
    
   
    words = query.lower().split()
    content_words = [w for w in words 
                     if w not in ['der', 'die', 'das', 'ein', 'eine', 'mit', 
                                  'und', 'oder', 'für', 'via', 'einer', 
                                  'automatisches', 'deployment', 'wie', 'kann', 'ich']]
    
    
    for word in content_words[:3]:  
        variants.append(f"{word} installation")
        variants.append(f"{word} tutorial")
        variants.append(f"{word} configuration")
    
    
    if 'copier' in content_words:
        variants.extend([
            "copier copy gitlab",
            "pip install copier",
            "git clone copier template"
        ])
    
    if 'shiny' in content_words:
        variants.append("shiny app deployment")
    
    if 'ollama' in content_words:
        variants.extend([
            "ollama gpu deployment",
            "ollama kubernetes yaml"
        ])
    
    return variants[:10]  


def multi_query_search(query: str, k_per_query: int = 5, final_k: int = 5) -> list:
    """
    Multi-Query-Suche mit query-relevantem Re-Ranking.
    
    
    Args:
        query: Original-Query
        k_per_query: Ergebnisse pro Variante
        final_k: Finale Top-K Ergebnisse
    
    Returns:
        Liste von Documents, sortiert nach Relevanz
    """
    global search_metadata
    import time
    
    start_time = time.time()
    
    
    reset_search_metadata()
    
    variants = generate_query_variants(query)
    
    
    search_metadata['num_query_variants'] = len(variants)
    search_metadata['query_variants'] = variants
    
    seen_content = set()
    all_results = []
    total_chunks_evaluated = 0
    
    for variant in variants:
        try:
            
            if hasattr(rag, 'search') and 'use_smart_search' in rag.search.__code__.co_varnames:
                results = rag.search(variant, k=k_per_query, use_smart_search=True)
            else:
                results = rag.search(variant, k=k_per_query)
            
            total_chunks_evaluated += len(results)
            
            for doc in results:
                
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_results.append(doc)
        except Exception as e:
            print(f"   Query-Variante failed: {variant[:30]}... ({e})")
            continue
    
    search_metadata['total_chunks_evaluated'] = total_chunks_evaluated
    search_metadata['unique_chunks_found'] = len(all_results)
    
    
    scored = []
    for doc in all_results:
        score = score_chunk(doc.page_content, query, doc.metadata)
        scored.append((score, doc))
    
    
    scored.sort(key=lambda x: -x[0])
    
    
    search_metadata['top_scores'] = [round(score, 1) for score, _ in scored[:final_k]]
    search_metadata['final_chunks_returned'] = min(len(scored), final_k)
    
    
    search_metadata['search_duration_ms'] = round((time.time() - start_time) * 1000, 1)
    
    return [doc for score, doc in scored[:final_k]]


@tool
def search_documentation(query: str) -> str:
    """
    Durchsucht die OpenShift Best Practices Dokumentation.
    
    Features:
    - Graph-RAG für bessere semantische Suche
    - Multi-Query mit Varianten
    - Query-relevantes Re-Ranking
    - Command-Extraction
    - Metadata-enhanced Scoring
    
    Args:
        query: Suchanfrage (z.B. "Wie deploye ich Ollama mit GPU?")
    
    Returns:
        Formatierte Suchergebnisse mit Befehlen
    """
    try:
        
        results = multi_query_search(query, k_per_query=5, final_k=5)
        
        if not results:
            return " Keine relevante Dokumentation gefunden."
        
        output = []
        output.append("=" * 70)
        output.append(f" SUCHERGEBNISSE: {query}")
        output.append("=" * 70)
        output.append("")
        output.append("WICHTIG: Übernimm die Befehle EXAKT wie dokumentiert!")
        output.append("")
        
        all_commands = []
        
        for i, doc in enumerate(results, 1):
            
            source = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', '?')
            
            
            llm_category = doc.metadata.get('llm_category', '')
            llm_difficulty = doc.metadata.get('llm_difficulty', '')
            llm_keywords_str = doc.metadata.get('llm_keywords', '')
            
            content = doc.page_content.strip()
            
            
            commands = extract_commands_from_text(content)
            all_commands.extend(commands)
            
            
            score = score_chunk(content, query, doc.metadata)
            
            
            output.append(f"--- [{i}] {source[:50]} (Seite {page}) ---")
            
            
            meta_info = []
            if llm_category:
                meta_info.append(f"Category: {llm_category}")
            if llm_difficulty:
                meta_info.append(f"Level: {llm_difficulty}")
            if meta_info:
                output.append(f"    {' | '.join(meta_info)}")
            
            output.append(f"    Score: {score:.0f} | Befehle: {len(commands)}")
            
            
            if llm_keywords_str:
                keywords = [k.strip() for k in llm_keywords_str.split(',')][:5]
                output.append(f"    Keywords: {', '.join(keywords)}")
            
            output.append("")
            
            
            display = content[:1500] if len(content) > 1500 else content
            output.append(display)
            output.append("")
        

        query_words = [w.lower() for w in query.split() 
                       if w.lower() not in ['der', 'die', 'das', 'ein', 'eine', 
                                            'mit', 'und', 'oder', 'für', 'via', 
                                            'einer', 'automatisches', 'deployment',
                                            'wie', 'kann', 'ich']]
        
        
        relevant_cmds = []
        other_cmds = []
        for cmd in all_commands:
            if any(word in cmd.lower() for word in query_words):
                relevant_cmds.append(cmd)
            else:
                other_cmds.append(cmd)
        
        unique_commands = list(dict.fromkeys(relevant_cmds + other_cmds))
        
        if unique_commands:
            output.append("=" * 70)
            output.append(" DOKUMENTIERTE BEFEHLE (exakt übernehmen!):")
            output.append("=" * 70)
            for i, cmd in enumerate(unique_commands[:20], 1):
                
                is_relevant = any(word in cmd.lower() for word in query_words)
                marker = "★" if is_relevant else " "
                output.append(f"  {marker}{i:2d}. {cmd}")
        else:
            output.append("")
            output.append("KEINE BEFEHLE IN DEN ERGEBNISSEN GEFUNDEN")
            output.append("(Möglicherweise nur konzeptuelle Dokumentation)")
        
        output.append("")
        output.append("=" * 70)
        
        return "\n".join(output)
        
    except Exception as e:
        import traceback
        return f"Fehler: {str(e)}\n\n{traceback.format_exc()}"


@tool
def get_learning_path(target_topic: str) -> str:
    """
    Erstellt einen vollständigen Lernpfad zu einem Ziel-Topic.
    Nutzt topologische Sortierung für optimale Reihenfolge.
    
    Args:
        target_topic: Ziel (z.B. "ollama-deployment", "gpu-workloads")
    
    Returns:
        Strukturierter Lernpfad mit allen Prerequisites
    """
    try:
        path = tutorial_nav.get_learning_path(target_topic)
        visualization = tutorial_nav.visualize_path(path)
        
        return f"""
Lernpfad zu '{target_topic}' erstellt!

{visualization}

HINWEIS: Dieser Pfad wurde durch topologische Sortierung optimiert.
Alle Prerequisites werden automatisch VOR dem Ziel-Topic behandelt.
"""
    except ValueError as e:
        
        all_topics = list(tutorial_nav.dependencies.keys())
        return f"""
 Fehler: {str(e)}

Verfügbare Topics:
{chr(10).join(f"  • {t}" for t in sorted(all_topics))}
"""


@tool
def list_available_topics(category: str = "all") -> str:
    """
    Listet alle verfügbaren Tutorial-Topics.
    
    Args:
        category: Filter (all, fundamentals, deployment, security, ml-ops, 
                  networking, operations, storage)
    
    Returns:
        Liste aller Topics mit Metadaten
    """
    all_topics = tutorial_nav.topological_sort()
    
    output = ["\n VERFÜGBARE TUTORIAL TOPICS\n" + "="*60 + "\n"]
    
    
    by_level = {'beginner': [], 'intermediate': [], 'advanced': []}
    
    for topic in all_topics:
        info = tutorial_nav.get_topic_info(topic)
        meta = info['metadata']
        
        
        if category != "all" and meta.get('category') != category:
            continue
        
        level = meta.get('level', 'unknown')
        by_level[level].append((topic, info))
    
    
    for level, icon in [('beginner', '🟢'), ('intermediate', '🟡'), ('advanced', '🔴')]:
        if not by_level[level]:
            continue
        
        output.append(f"\n{icon} {level.upper()}")
        output.append("-"*60)
        
        for topic, info in by_level[level]:
            meta = info['metadata']
            deps_count = len(info['dependencies'])
            
            output.append(f"\n  • {topic}")
            output.append(f"    {meta.get('description', 'N/A')}")
            output.append(f"    Category: {meta.get('category', 'N/A')} | Time: {meta.get('estimated_time', 'N/A')}")
            if deps_count > 0:
                output.append(f"    Prerequisites: {deps_count} topic(s)")
    
    output.append("\n" + "="*60)
    output.append(f"\nTotal Topics: {len(all_topics)}")
    
    if category != "all":
        output.append(f"Filter: {category}")
    
    return "\n".join(output)


@tool
def generate_tutorial_with_sources(topic: str, include_prerequisites: bool = True) -> str:
    """
    Generiert ein Tutorial mit explizitem Source-Tracking.
    
    Zeigt transparent:
    - Was kommt aus dem Navigator (Struktur)
    - Was kommt aus der Knowledge Base (Facts)
    
    Args:
        topic: Topic Identifier (z.B. "kubernetes-basics")
        include_prerequisites: Ob Prerequisites gezeigt werden
    
    Returns:
        Tutorial mit Content-Source Attribution
    """
    try:
        
        info = tutorial_nav.get_topic_info(topic)
        meta = info['metadata']
        
        
        search_queries = [
            f"{topic} guide tutorial",
            f"{topic} example deployment",
            f"{topic} best practices"
        ]
        
        all_docs = []
        for query in search_queries:
            docs = rag.search(query, k=2)
            all_docs.extend(docs)
        
        
        unique_docs = {doc.metadata.get('source'): doc for doc in all_docs}
        kb_docs = list(unique_docs.values())
        
        
        output = []
        output.append("="*80)
        output.append(f" TUTORIAL: {topic}")
        output.append("="*80)
        output.append("")
        
        
        output.append(" METADATA [Source: Tutorial Navigator]")
        output.append("-"*80)
        output.append(f"Level: {meta.get('level', 'N/A').upper()}")
        output.append(f"Category: {meta.get('category', 'N/A')}")
        output.append(f"Estimated Time: {meta.get('estimated_time', 'N/A')}")
        output.append(f"Description: {meta.get('description', 'N/A')}")
        output.append("")
        
       
        if info['dependencies'] and include_prerequisites:
            output.append(f" PREREQUISITES [Source: Tutorial Navigator]")
            output.append("-"*80)
            for dep in info['dependencies']:
                dep_meta = tutorial_nav.topic_metadata.get(dep, {})
                output.append(f"  ✓ {dep} ({dep_meta.get('level', 'N/A')})")
            output.append("")
        
        
        if kb_docs:
            output.append(" DOKUMENTIERTE INHALTE [Source: Knowledge Base]")
            output.append("-"*80)
            for i, doc in enumerate(kb_docs[:3], 1):
                source = doc.metadata.get('source', 'Unknown')
                
                
                llm_category = doc.metadata.get('llm_category', '')
                llm_difficulty = doc.metadata.get('llm_difficulty', '')
                
                content_preview = doc.page_content[:400].replace('\n', ' ').strip()
                
                output.append(f"\n[{i}] {source}")
                if llm_category:
                    output.append(f"    Category: {llm_category} | Difficulty: {llm_difficulty}")
                output.append(f"    Preview: {content_preview}...")
                
                
                if "```" in doc.page_content:
                    output.append(f"    Contains: Code examples ✓")
            
            if len(kb_docs) > 3:
                output.append(f"\n... and {len(kb_docs) - 3} more sources")
        else:
            output.append(" KNOWLEDGE BASE CONTENT")
            output.append("-"*80)
            output.append("  No relevant content found in Knowledge Base!")
            output.append("Tutorial would be synthesized from LLM training data.")
        
        output.append("")
        output.append("="*80)
        
        return "\n".join(output)
        
    except ValueError as e:
        return f" Error: {str(e)}"


@tool 
def calculate_kb_coverage_score(topic: str) -> str:
    """
    Berechnet einen Knowledge Base Coverage Score.
    
    Zeigt ob genug Dokumentation für ein Topic vorhanden ist.
    
    Args:
        topic: Topic zu evaluieren
    
    Returns:
        Detaillierter Coverage Report
    """
    try:
        
        docs = rag.search(f"{topic}", k=5)
        
       
        has_yaml = any('```yaml' in d.page_content for d in docs)
        has_commands = any(extract_commands_from_text(d.page_content) for d in docs)
        has_multiple_sources = len(set(d.metadata.get('source') for d in docs)) >= 2
        
        
        score = (
            (len(docs) / 5) * 0.4 +  
            (0.3 if has_yaml else 0) +  
            (0.2 if has_commands else 0) +  
            (0.1 if has_multiple_sources else 0)  
        )
        
        
        output = [
            f"\n KB COVERAGE REPORT: {topic}",
            "="*60,
            f"Overall Score: {score:.0%}",
            "",
            "Findings:",
            f"  {'✓' if len(docs) >= 3 else '✗'} Relevant Documents: {len(docs)}/5",
            f"  {'✓' if has_yaml else '✗'} YAML Examples: {has_yaml}",
            f"  {'✓' if has_commands else '✗'} Commands/Code: {has_commands}",
            f"  {'✓' if has_multiple_sources else '✗'} Multiple Sources: {has_multiple_sources}",
            "",
            "Recommendation:",
        ]
        
        if score >= 0.8:
            output.append("   EXCELLENT: KB has comprehensive content")
        elif score >= 0.6:
            output.append("   GOOD: Solid foundation, minor gaps")
        elif score >= 0.4:
            output.append("   MODERATE: Significant gaps in coverage")
        else:
            output.append("   INSUFFICIENT: Enrich KB before tutorial generation")
        
        output.append("="*60)
        
        return "\n".join(output)
        
    except Exception as e:
        return f" Error: {str(e)}"


@tool
def check_prerequisites(topic: str, completed_topics: str) -> str:
    """
    Prüft ob alle Prerequisites für ein Topic erfüllt sind.
    
    Args:
        topic: Ziel-Topic
        completed_topics: Komma-separierte Liste abgeschlossener Topics
    
    Returns:
        Status-Report mit fehlenden Prerequisites
    """
    try:
        info = tutorial_nav.get_topic_info(topic)
        completed = set(t.strip() for t in completed_topics.split(',') if t.strip())
        
        required = set(info['dependencies'])
        missing = required - completed
        
        output = [
            f"\n PREREQUISITE CHECK: {topic}",
            "="*60,
            f"Required Prerequisites: {len(required)}",
            f"Completed: {len(required) - len(missing)}/{len(required)}"
        ]
        
        if missing:
            output.append(f"\n MISSING PREREQUISITES ({len(missing)}):")
            for miss in sorted(missing):
                miss_meta = tutorial_nav.topic_metadata.get(miss, {})
                output.append(f"  • {miss} ({miss_meta.get('level', 'N/A')})")
            
            output.append(f"\n RECOMMENDATION:")
            output.append(f"   Use get_learning_path('{topic}') for full path")
        else:
            output.append(f"\n ALL PREREQUISITES FULFILLED!")
            output.append(f"   You can start with '{topic}'!")
        
        output.append("="*60)
        
        return "\n".join(output)
        
    except ValueError as e:
        return f" Error: {str(e)}"



@tool
def generate_deployment_yaml(
    app_name: str, 
    image: str, 
    port: int, 
    namespace: str = "default",
    memory: str = "512Mi", 
    cpu: str = "500m"
) -> str:
    """
    Generiert ein Kubernetes Deployment YAML.
    
    Args:
        app_name: Name der Applikation
        image: Container Image
        port: Port der Applikation
        namespace: Ziel-Namespace
        memory: Memory Limit
        cpu: CPU Limit
    
    Returns:
        YAML als String
    """
    
    deployment = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': app_name,
            'namespace': namespace,
            'labels': {
                'app': app_name,
                'created-by': 'ai-agent'
            }
        },
        'spec': {
            'replicas': 2,
            'selector': {
                'matchLabels': {
                    'app': app_name
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': app_name
                    }
                },
                'spec': {
                    'securityContext': {
                        'runAsNonRoot': True,
                        'seccompProfile': {
                            'type': 'RuntimeDefault'
                        }
                    },
                    'containers': [{
                        'name': app_name,
                        'image': image,
                        'ports': [{
                            'containerPort': port,
                            'protocol': 'TCP'
                        }],
                        'resources': {
                            'requests': {
                                'memory': memory,
                                'cpu': cpu
                            },
                            'limits': {
                                'memory': memory,
                                'cpu': cpu
                            }
                        },
                        'securityContext': {
                            'allowPrivilegeEscalation': False,
                            'capabilities': {
                                'drop': ['ALL']
                            }
                        },
                        'readinessProbe': {
                            'httpGet': {
                                'path': '/health',
                                'port': port
                            },
                            'initialDelaySeconds': 10,
                            'periodSeconds': 5
                        }
                    }]
                }
            }
        }
    }
    
    yaml_content = yaml.dump(deployment, default_flow_style=False)
    
    return f"""Deployment YAML generiert für '{app_name}' in namespace '{namespace}':

```yaml
{yaml_content}
```

Nächste Schritte:
1. Review das YAML
2. Deploy mit: deploy_application(yaml_content, "{namespace}", dry_run=True)
"""


@tool
def generate_advanced_deployment(
    app_name: str,
    image: str,
    port: int,
    namespace: str = "default",
    app_type: str = "generic",
    memory: str = "512Mi",
    cpu: str = "500m",
    replicas: int = 2,
    needs_storage: bool = False,
    storage_size: str = "10Gi",
    gpu_count: int = 0
) -> str:
    """
    Generiert ein Advanced Deployment mit mehr Konfigurationsoptionen.
    
    Args:
        app_name: Name der Applikation
        image: Container Image
        port: Application Port
        namespace: Ziel-Namespace
        app_type: Typ (generic/flask/fastapi/jupyter/ollama)
        memory: Memory Limit
        cpu: CPU Limit
        replicas: Anzahl Replicas
        needs_storage: Ob PVC benötigt wird
        storage_size: Größe des PVC
        gpu_count: Anzahl GPUs (0 = keine)
    
    Returns:
        YAML mit optionalem PVC
    """
    
    health_paths = {
        'flask': '/health',
        'fastapi': '/docs',
        'jupyter': '/lab',
        'generic': '/health',
        'ollama': None
    }
    
    health_path = health_paths.get(app_type, '/health')
    
    container = {
        'name': app_name,
        'image': image,
        'imagePullPolicy': 'Always',
        'ports': [{
            'containerPort': port,
            'protocol': 'TCP'
        }],
        'resources': {
            'requests': {
                'cpu': cpu,
                'memory': memory
            },
            'limits': {
                'cpu': cpu,
                'memory': memory
            }
        },
        'securityContext': {
            'allowPrivilegeEscalation': False,
            'capabilities': {
                'drop': ['ALL']
            }
        }
    }
    
    if gpu_count > 0:
        container['resources']['requests']['nvidia.com/gpu'] = str(gpu_count)
        container['resources']['limits']['nvidia.com/gpu'] = str(gpu_count)
        container['env'] = [{
            'name': 'CUDA_VISIBLE_DEVICES',
            'value': '0'
        }]
    
    if health_path:
        container['readinessProbe'] = {
            'httpGet': {
                'path': health_path,
                'port': port
            },
            'initialDelaySeconds': 10,
            'periodSeconds': 5
        }
    
    volumes = []
    volume_mounts = []
    pvcs = []
    
    if needs_storage:
        pvc_name = f"{app_name}-data"
        volumes.append({
            'name': 'app-data',
            'persistentVolumeClaim': {
                'claimName': pvc_name
            }
        })
        volume_mounts.append({
            'name': 'app-data',
            'mountPath': '/data'
        })
        
        pvcs.append({
            'apiVersion': 'v1',
            'kind': 'PersistentVolumeClaim',
            'metadata': {
                'name': pvc_name,
                'namespace': namespace
            },
            'spec': {
                'accessModes': ['ReadWriteOnce'],
                'resources': {
                    'requests': {
                        'storage': storage_size
                    }
                }
            }
        })
    
    if volume_mounts:
        container['volumeMounts'] = volume_mounts
    
    deployment = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': app_name,
            'namespace': namespace,
            'labels': {
                'app': app_name,
                'app-type': app_type,
                'created-by': 'ai-agent'
            }
        },
        'spec': {
            'replicas': replicas,
            'selector': {
                'matchLabels': {
                    'app': app_name
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': app_name
                    }
                },
                'spec': {
                    'containers': [container],
                    'securityContext': {
                        'runAsNonRoot': True,
                        'seccompProfile': {
                            'type': 'RuntimeDefault'
                        }
                    }
                }
            },
            'strategy': {
                'type': 'RollingUpdate',
                'rollingUpdate': {
                    'maxUnavailable': '25%',
                    'maxSurge': '25%'
                }
            }
        }
    }
    
    if volumes:
        deployment['spec']['template']['spec']['volumes'] = volumes
    
    output = f"Advanced Deployment generiert für '{app_name}' (Type: {app_type})\n\n"
    
    if pvcs:
        for pvc in pvcs:
            output += "**PersistentVolumeClaim:**\n```yaml\n"
            output += yaml.dump(pvc, default_flow_style=False)
            output += "```\n\n"
    
    output += "**Deployment:**\n```yaml\n"
    output += yaml.dump(deployment, default_flow_style=False)
    output += "```\n"
    
    return output


@tool
def generate_ollama_deployment(
    app_name: str,
    namespace: str,
    gpu_count: int = 1,
    memory: str = "32Gi",
    cpu: str = "8",
    pvc_name: str = None
) -> str:
    """
    Generiert ein Production-Ready Ollama Deployment mit GPU Support.
    
    Args:
        app_name: Name der Applikation
        namespace: Ziel-Namespace
        gpu_count: Anzahl GPUs
        memory: Memory Limit
        cpu: CPU Limit
        pvc_name: Name des PVC (optional)
    
    Returns:
        YAML mit PVC und Deployment
    """
    
    if pvc_name is None:
        pvc_name = app_name
    
    pvc = {
        'apiVersion': 'v1',
        'kind': 'PersistentVolumeClaim',
        'metadata': {
            'name': pvc_name,
            'namespace': namespace
        },
        'spec': {
            'accessModes': ['ReadWriteOnce'],
            'resources': {
                'requests': {
                    'storage': '50Gi'
                }
            }
        }
    }
    
    deployment = {
        'kind': 'Deployment',
        'apiVersion': 'apps/v1',
        'metadata': {
            'name': app_name,
            'namespace': namespace,
            'labels': {
                'app': app_name,
                'created-by': 'ai-agent'
            }
        },
        'spec': {
            'replicas': 1,
            'selector': {
                'matchLabels': {
                    'app': app_name
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': app_name
                    }
                },
                'spec': {
                    'volumes': [{
                        'name': 'ollama-data',
                        'persistentVolumeClaim': {
                            'claimName': pvc_name
                        }
                    }],
                    'containers': [{
                        'name': app_name,
                        'image': 'ollama/ollama:latest',
                        'imagePullPolicy': 'Always',
                        'ports': [{
                            'containerPort': 11434,
                            'protocol': 'TCP'
                        }],
                        'env': [{
                            'name': 'CUDA_VISIBLE_DEVICES',
                            'value': '0'
                        }],
                        'resources': {
                            'requests': {
                                'cpu': '1',
                                'memory': '2Gi',
                                'nvidia.com/gpu': str(gpu_count)
                            },
                            'limits': {
                                'cpu': cpu,
                                'memory': memory,
                                'nvidia.com/gpu': str(gpu_count)
                            }
                        },
                        'volumeMounts': [{
                            'name': 'ollama-data',
                            'mountPath': '/.ollama'
                        }],
                        'securityContext': {
                            'allowPrivilegeEscalation': False,
                            'capabilities': {
                                'drop': ['ALL']
                            }
                        },
                        'terminationMessagePath': '/dev/termination-log',
                        'terminationMessagePolicy': 'File'
                    }],
                    'restartPolicy': 'Always',
                    'terminationGracePeriodSeconds': 30,
                    'dnsPolicy': 'ClusterFirst',
                    'securityContext': {
                        'runAsNonRoot': True,
                        'seccompProfile': {
                            'type': 'RuntimeDefault'
                        }
                    },
                    'schedulerName': 'default-scheduler'
                }
            },
            'strategy': {
                'type': 'RollingUpdate',
                'rollingUpdate': {
                    'maxUnavailable': '25%',
                    'maxSurge': '25%'
                }
            },
            'revisionHistoryLimit': 10,
            'progressDeadlineSeconds': 600
        }
    }
    
    pvc_yaml = yaml.dump(pvc, default_flow_style=False)
    deployment_yaml = yaml.dump(deployment, default_flow_style=False)
    
    return f"""Production-Ready Ollama Deployment generiert!

**PersistentVolumeClaim für Model Storage:**
```yaml
{pvc_yaml}
```

**Deployment mit GPU Support:**
```yaml
{deployment_yaml}
```

Features:
- GPU Support ({gpu_count}x nvidia.com/gpu)
- Persistent Storage (50Gi PVC)
- Production Resources (Request: 2Gi, Limit: {memory})
- CUDA Environment configured
- Rolling Update Strategy

WICHTIG: Dein Cluster braucht GPU-Nodes mit nvidia.com/gpu Resource!

Nächste Schritte:
1. Erst PVC deployen: deploy_application(pvc_yaml, "{namespace}")
2. Dann Deployment: deploy_application(deployment_yaml, "{namespace}")
"""


@tool
def get_pod_status(namespace: str, app_name: str = None) -> str:
    """
    Holt den Status aller Pods in einem Namespace.
    
    Args:
        namespace: OpenShift Namespace
        app_name: Optional - filter nach app label
    
    Returns:
        Pod Status Report
    """
    
    try:
        cmd = ["oc", "get", "pods", "-n", namespace, "-o", "json"]
        
        if app_name:
            cmd.extend(["-l", f"app={app_name}"])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return f"Fehler beim Abrufen der Pods: {result.stderr}"
        
        pods_data = json.loads(result.stdout)
        pods = pods_data.get('items', [])
        
        if not pods:
            return f" Keine Pods gefunden in namespace '{namespace}'" + (f" mit app={app_name}" if app_name else "")
        
        output = [f"Pod Status in '{namespace}':\n"]
        
        for pod in pods:
            name = pod['metadata']['name']
            status = pod['status']['phase']
            
            container_statuses = pod['status'].get('containerStatuses', [])
            ready = "ok" if all(c.get('ready', False) for c in container_statuses) else "❌"
            
            restart_count = sum(c.get('restartCount', 0) for c in container_statuses)
            
            output.append(f"{ready} {name}")
            output.append(f"   Status: {status}")
            output.append(f"   Restarts: {restart_count}")
            
            if status != "Running":
                for container in container_statuses:
                    if 'state' in container:
                        state = container['state']
                        if 'waiting' in state:
                            reason = state['waiting'].get('reason', 'Unknown')
                            output.append(f"   Waiting: {reason}")
            output.append("")
        
        return "\n".join(output)
        
    except subprocess.TimeoutExpired:
        return "Timeout beim Abrufen der Pods"
    except json.JSONDecodeError:
        return f"Fehler beim Parsen der oc-Ausgabe: {result.stdout}"
    except Exception as e:
        return f"Unerwarteter Fehler: {str(e)}"


@tool
def deploy_application(yaml_content: str, namespace: str, dry_run: bool = True) -> str:
    """
    Deployed eine Applikation in OpenShift.
    
    Args:
        yaml_content: Das Deployment YAML als String
        namespace: Ziel-Namespace
        dry_run: Wenn True, nur Validierung (empfohlen!)
    
    Returns:
        Deployment Status
    """
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name
        
        cmd = ["oc", "apply", "-f", temp_file, "-n", namespace]
        
        if dry_run:
            cmd.append("--dry-run=server")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        import os
        os.unlink(temp_file)
        
        if result.returncode != 0:
            return f"Deployment fehlgeschlagen:\n{result.stderr}"
        
        if dry_run:
            return f"""Dry-run erfolgreich! YAML ist valid.

{result.stdout}

NICHT deployed (dry_run=True).
Um wirklich zu deployen: deploy_application(yaml_content, namespace, dry_run=False)
"""
        else:
            return f"""Deployment erfolgreich!

{result.stdout}

Prüfe Status mit: get_pod_status('{namespace}')
"""
        
    except subprocess.TimeoutExpired:
        return "Timeout beim Deployment"
    except Exception as e:
        return f"Fehler: {str(e)}"
