from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque
import json


class TutorialNavigator:
    """
    Verwaltet Tutorial-Themen und ihre Abhängigkeiten.
    Nutzt topologische Sortierung für optimale Lernreihenfolge.
    """
    
    def __init__(self):
       
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        
        self.topic_metadata: Dict[str, dict] = {}
        
       
        self._initialize_default_topics()
    
    def _initialize_default_topics(self):
        """
        Initialisiert Standard DevOps/Kubernetes Topics mit Dependencies.
        
        Struktur: Foundations -> Intermediate -> Advanced
        """
        
        
        self.add_topic(
            "kubernetes-basics",
            dependencies=[],
            metadata={
                "level": "beginner",
                "category": "fundamentals",
                "description": "Grundlagen: Pods, Services, Deployments",
                "estimated_time": "30min"
            }
        )
        
        self.add_topic(
            "container-basics",
            dependencies=[],
            metadata={
                "level": "beginner",
                "category": "fundamentals",
                "description": "Container, Images, Docker/Podman Basics",
                "estimated_time": "20min"
            }
        )
        
        self.add_topic(
            "yaml-basics",
            dependencies=[],
            metadata={
                "level": "beginner",
                "category": "fundamentals",
                "description": "YAML Syntax und Best Practices",
                "estimated_time": "15min"
            }
        )
        
       
        self.add_topic(
            "simple-deployment",
            dependencies=["kubernetes-basics", "container-basics", "yaml-basics"],
            metadata={
                "level": "beginner",
                "category": "deployment",
                "description": "Einfache Applikation deployen",
                "estimated_time": "45min"
            }
        )
        
        self.add_topic(
            "networking-basics",
            dependencies=["kubernetes-basics"],
            metadata={
                "level": "beginner",
                "category": "networking",
                "description": "Services, ClusterIP, NodePort",
                "estimated_time": "30min"
            }
        )
        
        
        self.add_topic(
            "resource-management",
            dependencies=["simple-deployment"],
            metadata={
                "level": "intermediate",
                "category": "operations",
                "description": "CPU/Memory Requests & Limits",
                "estimated_time": "40min"
            }
        )
        
        self.add_topic(
            "persistent-storage",
            dependencies=["simple-deployment"],
            metadata={
                "level": "intermediate",
                "category": "storage",
                "description": "PVCs, Volumes, StatefulSets",
                "estimated_time": "50min"
            }
        )
        
        self.add_topic(
            "health-checks",
            dependencies=["simple-deployment"],
            metadata={
                "level": "intermediate",
                "category": "operations",
                "description": "Liveness & Readiness Probes",
                "estimated_time": "35min"
            }
        )
        
        self.add_topic(
            "security-basics",
            dependencies=["simple-deployment"],
            metadata={
                "level": "intermediate",
                "category": "security",
                "description": "SecurityContext, RunAsNonRoot, Capabilities",
                "estimated_time": "45min"
            }
        )
        
        
        self.add_topic(
            "gpu-workloads",
            dependencies=["resource-management", "simple-deployment"],
            metadata={
                "level": "advanced",
                "category": "ml-ops",
                "description": "GPU Scheduling & Resource Limits",
                "estimated_time": "60min"
            }
        )
        
        self.add_topic(
            "ollama-deployment",
            dependencies=["gpu-workloads", "persistent-storage", "security-basics"],
            metadata={
                "level": "advanced",
                "category": "ml-ops",
                "description": "LLM Server mit GPU und Storage",
                "estimated_time": "90min"
            }
        )
        
        self.add_topic(
            "advanced-networking",
            dependencies=["networking-basics", "security-basics"],
            metadata={
                "level": "advanced",
                "category": "networking",
                "description": "Ingress, NetworkPolicies, Service Mesh",
                "estimated_time": "75min"
            }
        )
        
        self.add_topic(
            "production-best-practices",
            dependencies=[
                "resource-management",
                "health-checks",
                "security-basics",
                "persistent-storage"
            ],
            metadata={
                "level": "advanced",
                "category": "operations",
                "description": "Production-Ready Deployments",
                "estimated_time": "120min"
            }
        )
    
    def add_topic(self, topic: str, dependencies: List[str], metadata: dict = None):
        """
        Fügt ein Topic mit seinen Dependencies hinzu.
        
        Args:
            topic: Topic Identifier
            dependencies: Liste von Topics, die vorher gelernt werden müssen
            metadata: Zusätzliche Informationen (Level, Category, etc.)
        """
        self.dependencies[topic] = set(dependencies)
        self.topic_metadata[topic] = metadata or {}
    
    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """Gibt den kompletten Dependency Graph zurück"""
        return dict(self.dependencies)
    
    def topological_sort(self, topics: List[str] = None) -> List[str]:
        """
        Führt topologische Sortierung durch (Kahn's Algorithm).
        
        Args:
            topics: Spezifische Topics zum Sortieren (None = alle)
        
        Returns:
            Liste von Topics in optimaler Lernreihenfolge
        
        Raises:
            ValueError: Wenn zyklische Dependencies existieren
        """
      
        if topics is None:
            topics = list(self.dependencies.keys())
        else:
           
            for topic in topics:
                if topic not in self.dependencies:
                    raise ValueError(f"Unknown topic: {topic}")
        
       
        in_degree = {topic: 0 for topic in topics}
        
        
        relevant_deps = {}
        for topic in topics:
            relevant = self.dependencies[topic] & set(topics)
            relevant_deps[topic] = relevant
        
        
        for topic in topics:
            for dep in relevant_deps[topic]:
                in_degree[topic] += 1
        
      
        queue = deque([topic for topic in topics if in_degree[topic] == 0])
        result = []
        
        while queue:
            
            current = queue.popleft()
            result.append(current)
            
            
            for topic in topics:
                if current in relevant_deps[topic]:
                    in_degree[topic] -= 1
                    if in_degree[topic] == 0:
                        queue.append(topic)
        
        
        if len(result) != len(topics):
            raise ValueError(
                f"Zyklische Dependencies gefunden! "
                f"Sortiert: {len(result)}, Erwartet: {len(topics)}"
            )
        
        return result
    
    def get_learning_path(self, target_topic: str) -> List[str]:
        """
        Erstellt einen vollständigen Lernpfad zu einem Target-Topic.
        Inkludiert automatisch alle Prerequisites.
        
        Args:
            target_topic: Ziel-Topic (z.B. "ollama-deployment")
        
        Returns:
            Sortierte Liste von Topics vom Foundation bis zum Target
        """
        if target_topic not in self.dependencies:
            raise ValueError(f"Unknown topic: {target_topic}")
        
        
        required_topics = self._collect_dependencies(target_topic)
        required_topics.add(target_topic)
        
        
        return self.topological_sort(list(required_topics))
    
    def _collect_dependencies(self, topic: str, visited: Set[str] = None) -> Set[str]:
        """Rekursive Helper-Funktion für Dependency Collection"""
        if visited is None:
            visited = set()
        
        if topic in visited:
            return visited
        
        visited.add(topic)
        
        for dep in self.dependencies.get(topic, set()):
            self._collect_dependencies(dep, visited)
        
        return visited
    
    def get_topic_info(self, topic: str) -> dict:
        """
        Gibt vollständige Informationen zu einem Topic zurück.
        """
        if topic not in self.dependencies:
            raise ValueError(f"Unknown topic: {topic}")
        
        return {
            "topic": topic,
            "dependencies": list(self.dependencies[topic]),
            "metadata": self.topic_metadata.get(topic, {}),
            "num_prerequisites": len(self.dependencies[topic])
        }
    
    def visualize_path(self, topics: List[str]) -> str:
        """
        Erstellt eine visuelle Darstellung eines Lernpfads.
        """
        output = []
        output.append("\n" + "="*80)
        output.append("TUTORIAL LEARNING PATH")
        output.append("="*80 + "\n")
        
        for i, topic in enumerate(topics, 1):
            info = self.get_topic_info(topic)
            meta = info['metadata']
            
            # Level Badge
            level = meta.get('level', 'unknown').upper()
            level_badge = {
                'BEGINNER': '🟢',
                'INTERMEDIATE': '🟡',
                'ADVANCED': '🔴'
            }.get(level, '⚪')
            
            output.append(f"Step {i}: {level_badge} {topic}")
            output.append(f"        Level: {meta.get('level', 'N/A')}")
            output.append(f"        Category: {meta.get('category', 'N/A')}")
            output.append(f"        Description: {meta.get('description', 'N/A')}")
            output.append(f"        Time: {meta.get('estimated_time', 'N/A')}")
            
            if info['dependencies']:
                output.append(f"        Prerequisites: {', '.join(info['dependencies'])}")
            
            output.append("")
        
        total_time = sum(
            int(self.topic_metadata.get(t, {}).get('estimated_time', '0').replace('min', ''))
            for t in topics
        )
        output.append(f"Total Estimated Time: {total_time} minutes (~{total_time/60:.1f} hours)")
        output.append("="*80 + "\n")
        
        return "\n".join(output)
    
    def get_next_topics(self, completed_topics: List[str]) -> List[str]:
        """
        Gibt Topics zurück, die als nächstes gelernt werden können.
        Berücksichtigt bereits abgeschlossene Topics.
        
        Args:
            completed_topics: Liste bereits gelernter Topics
        
        Returns:
            Liste von Topics, deren Prerequisites erfüllt sind
        """
        completed = set(completed_topics)
        available = []
        
        for topic in self.dependencies:
            if topic in completed:
                continue
            
           
            deps = self.dependencies[topic]
            if deps.issubset(completed):
                available.append(topic)
        
        return available
    
    def export_graph(self, filename: str = "tutorial_graph.json"):
        """Exportiert den Dependency Graph als JSON"""
        graph_data = {
            "topics": {
                topic: {
                    "dependencies": list(deps),
                    "metadata": self.topic_metadata.get(topic, {})
                }
                for topic, deps in self.dependencies.items()
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f" Graph exportiert nach: {filename}")



if __name__ == "__main__":
    print("\n" + "="*80)
    print("TUTORIAL NAVIGATOR DEMO")
    print("="*80 + "\n")
    
    navigator = TutorialNavigator()
    
    
    print(" DEMO 1: Lernpfad zu 'ollama-deployment'")
    print("-"*80)
    
    path = navigator.get_learning_path("ollama-deployment")
    print(navigator.visualize_path(path))
    
    
    print("\n DEMO 2: Alle Topics (vollständiger Curriculum)")
    print("-"*80)
    
    all_topics = navigator.topological_sort()
    print("\nOptimale Reihenfolge:")
    for i, topic in enumerate(all_topics, 1):
        level = navigator.topic_metadata[topic].get('level', 'N/A')
        print(f"  {i:2d}. [{level:12s}] {topic}")
    
  
    print("\n\n DEMO 3: Progressive Learning")
    print("-"*80)
    
    completed = ["kubernetes-basics", "container-basics", "yaml-basics"]
    print(f"Completed Topics: {', '.join(completed)}")
    
    next_available = navigator.get_next_topics(completed)
    print(f"\nNext Available Topics ({len(next_available)}):")
    for topic in next_available:
        meta = navigator.topic_metadata[topic]
        print(f"  • {topic} ({meta.get('level', 'N/A')})")
    
    
    print("\n\n DEMO 4: Custom Learning Path")
    print("-"*80)
    
    custom_topics = ["simple-deployment", "persistent-storage", "gpu-workloads", "ollama-deployment"]
    custom_path = navigator.topological_sort(custom_topics)
    print(f"\nCustom Path: {' → '.join(custom_path)}")
    
    
    print("\n\n DEMO 5: Export Graph")
    print("-"*80)
    navigator.export_graph("/mnt/user-data/outputs/tutorial_graph.json")
