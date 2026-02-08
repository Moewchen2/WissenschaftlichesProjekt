from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import sys
import json

from evaluation_config import TEST_CASES

sys.path.insert(0, '/mnt/user-data/outputs')
from agent_tools import (
    get_learning_path,
    list_available_topics,
    generate_tutorial_with_sources,
    calculate_kb_coverage_score,
    check_prerequisites,
    search_documentation,
    generate_deployment_yaml,
    generate_advanced_deployment,
    generate_ollama_deployment,
    get_pod_status,
    deploy_application,
    get_search_metadata,
    reset_search_metadata
)

SYSTEM_PROMPT = """Du bist ein Tutorial-Generator für die OpenShift-Umgebung.

## UMGEBUNG

Du erstellst Anleitungen für Workbenches - eine eingeschränkte Container-Umgebung:
- Kein Root-Zugriff
- Lokale S3-Endpoints (nicht AWS)
- Spezifische Installationspfade

## KERNREGEL: BEFEHLE EXAKT ÜBERNEHMEN

Wenn search_documentation() Befehle zurückgibt, übernimm sie ZEICHENGENAU.

**Beispiel:**

Dokumentation zeigt:
```
./aws/install -i /opt/app-root/src -b /opt/app-root/src
```

RICHTIG (exakt übernommen):
```bash
./aws/install -i /opt/app-root/src -b /opt/app-root/src
```

FALSCH (aus Training "verbessert"):
```bash
sudo ./aws/install
```

## ARBEITSWEISE

1. Rufe `search_documentation(query)` auf
2. Schau dir die Liste "DOKUMENTIERTE BEFEHLE" an
3. Übernimm diese Befehle EXAKT in deine Anleitung
4. Formuliere Erklärungen aus dem KONTEXT-Bereich

## AUSGABE-FORMAT

Schreibe natürliche, erklärende Anleitungen auf Deutsch (Sie-Form):

```
[Nr]. [Was wird gemacht und warum]:

   ```bash
   exakter-befehl-aus-dokumentation
   ```

   [Bei Prüfbefehlen: Erwartete Ausgabe]
   
   **Achtung:** [Warnung, falls in Doku erwähnt]
```

## STILREGELN

- Natürliches, flüssiges Deutsch
- Erklärung VOR dem Befehl (warum wird das gemacht?)
- Bei Prüfbefehlen: Was sollte man sehen?
- Warnungen aus der Dokumentation übernehmen
- Keine holprigen Formulierungen
- Keine übermäßige Formatierung

## BEISPIEL EINER GUTEN ANLEITUNG

```
1. Laden Sie zunächst die Installationsdatei der AWS CLI herunter:

   ```bash
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   ```

   **Achtung:** Die Proxy-Server-Variablen müssen eingerichtet sein.

2. Entpacken Sie die heruntergeladene Datei:

   ```bash
   unzip awscliv2.zip
   ```

3. Installieren Sie die AWS CLI in Ihrem Benutzerverzeichnis:

   ```bash
   ./aws/install -i /opt/app-root/src -b /opt/app-root/src
   ```

4. Um den `aws`-Befehl ohne vollständigen Pfad nutzen zu können, 
   hinterlegen Sie einen Alias:

   ```bash
   echo "alias aws='~/v2/current/bin/aws'" >> ~/.bashrc
   source ~/.bashrc
   ```

5. Prüfen Sie, ob die Installation erfolgreich war:

   ```bash
   aws --version
   ```

   Sie sollten eine Ausgabe wie `aws-cli/2.15.0 Python/3.11...` sehen.
```

## SELBSTPRÜFUNG VOR AUSGABE

- Steht JEDER Befehl exakt so in den Suchergebnissen?
- Habe ich Befehle aus meinem Training statt aus der Doku genutzt?
- Sind die Erklärungen natürlich formuliert?

## TOOLS

- `search_documentation(query)` - IMMER ZUERST, Befehle daraus EXAKT übernehmen
- `get_learning_path(topic)` - Lernpfade anzeigen
- `list_available_topics(category)` - Verfügbare Themen
- `generate_deployment_yaml(...)` - YAML generieren
"""

llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="XXX",
    timeout=120,
    temperature=0
)

tools = [
    get_learning_path,
    list_available_topics,
    generate_tutorial_with_sources,
    calculate_kb_coverage_score,
    check_prerequisites,
    search_documentation,
    generate_deployment_yaml,
    generate_advanced_deployment,
    generate_ollama_deployment,
    get_pod_status,
    deploy_application
]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)


SUSPICIOUS_PATTERNS = [
    ('sudo ', 'sudo existiert nicht in Workbenches'),
    ('apt-get ', 'Paketmanager nicht verfügbar'),
    ('yum ', 'Paketmanager nicht verfügbar'),
    ('pip install awscli', 'AWS CLI wird anders installiert'),
    ('--region eu-', 'Region nicht nötig bei PLAIN S3'),
    ('--region us-', 'Region nicht nötig bei PLAIN S3'),
]


def validate_output(output: str) -> list:
    """Prüft auf verdächtige Patterns und gibt Warnungen zurück."""
    warnings = []
    for pattern, message in SUSPICIOUS_PATTERNS:
        if pattern in output:
            warnings.append(f"Gefunden: '{pattern}' - {message}")
    return warnings



class TutorialAgent:
    def __init__(self):
        self.chat_history = []
        self.last_search_metadata = None 
    
    def chat(self, user_input: str) -> dict:
        import time
        start = time.time()
        
        
        reset_search_metadata()
    
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })
    
        
        result['latency'] = time.time() - start
        result['num_iterations'] = len(result.get('intermediate_steps', []))
        result['num_tool_calls'] = len(result.get('intermediate_steps', []))
        result['tools_used'] = [
           action.tool for action, _ in result.get('intermediate_steps', [])
        ]
        
       
        self.last_search_metadata = get_search_metadata()
        result['search_metadata'] = self.last_search_metadata
    
        
        result['warnings'] = validate_output(result['output'])
    
        
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=result['output']))
    
        return result
    
    def reset(self):
        self.chat_history = []
        self.last_search_metadata = None
        print("Konversation zurückgesetzt")
    
    def get_tool_calls(self, result: dict) -> list:
        """Gibt die verwendeten Tools zurück."""
        return [
            {'tool': action.tool, 'input': action.tool_input}
            for action, _ in result.get('intermediate_steps', [])
        ]




def run_interactive():
    print("\n" + "="*70)
    print("OpenShift Tutorial Agent")
    print("   Search Metadata Tracking für Evaluation")
    print("="*70)
    print("\nBefehle:")
    print("   exit/quit  - Beenden")
    print("   reset      - Konversation zurücksetzen")
    print("   tools      - Letzte Tool-Aufrufe anzeigen")
    print("   search     - Search-Metadaten anzeigen") 
    print("-"*70)
    print("\nBeispiele:")
    print("   • Wie installiere ich die AWS CLI?")
    print("   • Wie erstelle ich ein S3 Bucket?")
    print("   • Zeig mir den Lernpfad für Ollama")
    print("-"*70 + "\n")
    
    agent = TutorialAgent()
    last_result = None
    
    evaluation_log = []

    while True:
        try:
            user_input = input("\nDu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nTschüss!")
            break
        
        if not user_input:
            continue
        
        
        if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
            print("Tschüss!")
            break
        
        if user_input.lower() == 'reset':
            agent.reset()
            last_result = None
            continue
        
        if user_input.lower() == 'export':
            if evaluation_log:
                filename = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_log, f, indent=2, ensure_ascii=False)
                print(f"Exportiert: {filename}")
            else:
                print("Noch keine Daten zum Exportieren.")
            continue

        if user_input.lower() == 'tools':
            if last_result:
                tools_used = agent.get_tool_calls(last_result)
                if tools_used:
                    print("\nVerwendete Tools:")
                    for t in tools_used:
                        print(f"   → {t['tool']}({t['input']})")
                else:
                    print("Keine Tools verwendet.")
            else:
                print("Noch keine Anfrage gestellt.")
            continue
        
        # NEU: Search-Metadaten anzeigen
        if user_input.lower() == 'search':
            if last_result and last_result.get('search_metadata'):
                meta = last_result['search_metadata']
                print("\nSearch Metadata:")
                print(f"   Query-Varianten: {meta['num_query_variants']}")
                if meta['query_variants']:
                    print("   Varianten:")
                    for v in meta['query_variants'][:5]:
                        print(f"      • {v}")
                    if len(meta['query_variants']) > 5:
                        print(f"      ... und {len(meta['query_variants']) - 5} weitere")
                print(f"   Chunks evaluiert: {meta['total_chunks_evaluated']}")
                print(f"   Unique Chunks: {meta['unique_chunks_found']}")
                print(f"   Finale Chunks: {meta['final_chunks_returned']}")
                print(f"   Top-Scores: {meta['top_scores']}")
                print(f"   Such-Dauer: {meta['search_duration_ms']}ms")
            else:
                print("Keine Search-Metadaten verfügbar (kein search_documentation Aufruf).")
            continue

        if user_input.lower() == 'eval':
            print("\nStarte Evaluation Suite...")
            results = run_evaluation_suite(agent, TEST_CASES)
    
            
            filename = f"eval_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
            print_evaluation_summary(results)
            print(f"\n📁 Ergebnisse exportiert: {filename}")
            continue
        
        
        try:
            print("\nVerarbeite...\n")
            last_result = agent.chat(user_input)
            
           
            eval_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': user_input,
                'latency_seconds': round(last_result.get('latency'), 2),
                'tools_used': agent.get_tool_calls(last_result),
                'num_tool_calls': last_result.get('num_tool_calls', 0),
                'num_warnings': len(last_result.get('warnings', [])),
                'output_length': len(last_result.get('output', ''))
            }
            
            
            if last_result.get('search_metadata'):
                eval_entry['search_metadata'] = last_result['search_metadata']
            
            evaluation_log.append(eval_entry)
            
            
            print("="*70)
            print("Agent:")
            print("="*70)
            print(last_result['output'])
            
           
            if last_result.get('warnings'):
                print("\n" + "-"*70)
                print("VALIDIERUNG - Mögliche Probleme:")
                for w in last_result['warnings']:
                    print(f"   {w}")
                print("-"*70)
            
        except Exception as e:
            print(f"\nFehler: {str(e)}")
            import traceback
            traceback.print_exc()

def run_evaluation_suite(agent: TutorialAgent, test_cases: list) -> list:
    """
    Führt alle Testfälle automatisch aus.
    
    Args:
        agent: TutorialAgent-Instanz
        test_cases: Liste der Testfälle
    
    Returns:
        Liste mit Ergebnissen inkl. Search Metadata
    """
    from datetime import datetime
    results = []
    
    print("="*70)
    print("EVALUATION SUITE")
    print(f"Testfälle: {len(test_cases)}")
    print("="*70 + "\n")
    
    for i, tc in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {tc['id']}: {tc['query'][:50]}...")
        
        agent.reset() 
        
        try:
            result = agent.chat(tc['query'])
            
           
            tools_used = [t['tool'] for t in agent.get_tool_calls(result)]
            warnings = result.get('warnings', [])
            
            
            output = result.get('output', '')
            expected_keywords = tc.get('expected_keywords', [])
            keywords_found = [kw for kw in expected_keywords 
                  if kw.lower() in output.lower()]
            
            eval_result = {
                'test_id': tc['id'],
                'category': tc['category'],
                'timestamp': datetime.now().isoformat(),
                'query': tc['query'],
                'latency_seconds': round(result.get('latency', 0), 2),
                'tools_used': tools_used,
                'expected_tools': tc.get('expected_tools', []),
                'tools_match': set(tools_used) == set(tc.get('expected_tools', [])),
                'num_tool_calls': len(tools_used),
                'expected_keywords': expected_keywords,
                'keywords_found': keywords_found,
                'keywords_missing': [kw for kw in expected_keywords if kw.lower() not in output.lower()],
                'keywords_coverage': len(keywords_found) / len(expected_keywords) if expected_keywords else 1.0,
                'num_warnings': len(warnings),
                'warnings': warnings,
                'output_length': len(output),
                'success': True
            }
            
            
            if result.get('search_metadata'):
                sm = result['search_metadata']
                eval_result['search_metadata'] = {
                    'num_query_variants': sm['num_query_variants'],
                    'query_variants': sm['query_variants'],
                    'total_chunks_evaluated': sm['total_chunks_evaluated'],
                    'unique_chunks_found': sm['unique_chunks_found'],
                    'final_chunks_returned': sm['final_chunks_returned'],
                    'top_scores': sm['top_scores'],
                    'search_duration_ms': sm['search_duration_ms']
                }
            
            status = "✓" if eval_result['tools_match'] else "○"
            
            
            search_info = ""
            if result.get('search_metadata'):
                sm = result['search_metadata']
                search_info = f" | Variants: {sm['num_query_variants']} | Chunks: {sm['unique_chunks_found']}"
            
            print(f"   {status} Latenz: {eval_result['latency_seconds']}s | "
                  f"Tools: {tools_used}{search_info}")
            
        except Exception as e:
            eval_result = {
                'test_id': tc['id'],
                'category': tc['category'],
                'timestamp': datetime.now().isoformat(),
                'query': tc['query'],
                'error': str(e),
                'success': False
            }
            print(f"   ✗ Fehler: {str(e)[:50]}")
        
        results.append(eval_result)
        print()
    
    return results


def print_evaluation_summary(results: list):
    """Gibt eine Zusammenfassung der Evaluation aus (erweitert um Search Metadata)."""
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"\nGesamt: {len(results)} | Erfolg: {len(successful)} | Fehler: {len(failed)}")
    
    if successful:
        avg_latency = sum(r['latency_seconds'] for r in successful) / len(successful)
        avg_tools = sum(r['num_tool_calls'] for r in successful) / len(successful)
        tools_match = sum(1 for r in successful if r.get('tools_match', False))
        
        print(f"\nPerformance:")
        print(f"   Ø Latenz: {avg_latency:.2f}s")
        print(f"   Ø Tool-Calls: {avg_tools:.1f}")
        print(f"   Tool-Match: {tools_match}/{len(successful)}")
        
        
        with_search = [r for r in successful if r.get('search_metadata')]
        if with_search:
            avg_variants = sum(r['search_metadata']['num_query_variants'] for r in with_search) / len(with_search)
            avg_chunks = sum(r['search_metadata']['unique_chunks_found'] for r in with_search) / len(with_search)
            avg_search_ms = sum(r['search_metadata']['search_duration_ms'] for r in with_search) / len(with_search)
            
            print(f"\nSearch Metadata ({len(with_search)} Testfälle mit search_documentation):")
            print(f"   Ø Query-Varianten: {avg_variants:.1f}")
            print(f"   Ø Unique Chunks: {avg_chunks:.1f}")
            print(f"   Ø Such-Dauer: {avg_search_ms:.0f}ms")
    
    
    print(f"\nNach Kategorie:")
    for cat in ['simple', 'medium', 'complex', 'edge']:
        cat_results = [r for r in successful if r.get('category') == cat]
        if cat_results:
            avg = sum(r['latency_seconds'] for r in cat_results) / len(cat_results)
            print(f"   {cat}: {len(cat_results)} Tests | Ø {avg:.2f}s")
    
    print("="*70)



if __name__ == "__main__":
    run_interactive()
