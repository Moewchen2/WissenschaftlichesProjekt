TEST_CASES = [
    # =========================================================================
    # KATEGORIE 1: Einfach (1 Tool-Call erwartet)
    # =========================================================================
    {
        "id": "TC-01",
        "category": "simple",
        "query": "Wie installiere ich die AWS CLI?",
        "expected_tools": ["search_documentation"],
        "expected_keywords": ["curl", "unzip", "install"],
        "description": "Grundlegende Installation, lineare Schritte"
    },
    {
        "id": "TC-02",
        "category": "simple",
        "query": "Wie liste ich alle S3 Buckets auf?",
        "expected_tools": ["search_documentation"],
        "expected_keywords": ["aws s3 ls"],
        "description": "Einzelner Befehl aus Dokumentation"
    },
    {
        "id": "TC-03",
        "category": "simple",
        "query": "Welche Topics sind verfügbar?",
        "expected_tools": ["list_available_topics"],
        "expected_keywords": [],
        "description": "Reine Tool-Abfrage ohne RAG"
    },
    
    # =========================================================================
    # KATEGORIE 2: Mittel (1-2 Tool-Calls, mehrere Schritte)
    # =========================================================================
    {
        "id": "TC-04",
        "category": "medium",
        "query": "Wie erstelle ich Credentials in der Artifactory?",
        "expected_tools": ["search_documentation"],
        "expected_keywords": ["credentials", "token"],
        "description": "Mehrstufiger Prozess mit Konfiguration"
    },
    {
        "id": "TC-05",
        "category": "medium",
        "query": "Wie konfiguriere ich S3 mit Endpoint und Credentials?",
        "expected_tools": ["search_documentation"],
        "expected_keywords": ["aws/config", "aws/credentials", "endpoint_url"],
        "description": "Mehrere Konfigurationsdateien"
    },
    {
        "id": "TC-06",
        "category": "medium",
        "query": "Zeige mir den Lernpfad für GPU-Workloads",
        "expected_tools": ["get_learning_path"],
        "expected_keywords": ["gpu-workloads", "prerequisites"],
        "description": "Dependency-Aware Navigation"
    },
    {
        "id": "TC-07",
        "category": "medium",
        "query": "Generiere ein Deployment YAML für eine Flask-App auf Port 5000",
        "expected_tools": ["generate_deployment_yaml"],
        "expected_keywords": ["apiVersion", "kind: Deployment", "containerPort"],
        "description": "YAML-Generierung mit Parametern"
    },
    
    # =========================================================================
    # KATEGORIE 3: Komplex (2+ Tool-Calls, Dependencies)
    # =========================================================================
    {
        "id": "TC-08",
        "category": "complex",
        "query": "Wie deploye ich Ollama mit GPU-Unterstützung?",
        "expected_tools": ["search_documentation", "generate_ollama_deployment"],
        "expected_keywords": ["nvidia.com/gpu", "PersistentVolumeClaim"],
        "description": "RAG + YAML-Generierung kombiniert"
    },
    {
        "id": "TC-09",
        "category": "complex",
        "query": "Zeige mir den kompletten Lernpfad für Ollama-Deployment und prüfe die KB-Abdeckung",
        "expected_tools": ["get_learning_path", "calculate_kb_coverage_score"],
        "expected_keywords": ["ollama-deployment", "Coverage"],
        "description": "Mehrere Tools für Analyse"
    },
    {
        "id": "TC-10",
        "category": "complex",
        "query": "Erstelle ein Advanced Deployment für Jupyter mit GPU und persistentem Storage",
        "expected_tools": ["generate_advanced_deployment"],
        "expected_keywords": ["nvidia.com/gpu", "PersistentVolumeClaim", "volumeMounts"],
        "description": "Komplexe YAML-Generierung"
    },
    
    # =========================================================================
    # KATEGORIE 4: Edge Cases
    # =========================================================================
    {
        "id": "TC-11",
        "category": "edge",
        "query": "Wie installiere ich Software mit sudo?",
        "expected_tools": ["search_documentation"],
        "expected_warnings": ["sudo"],
        "description": "Sollte Warnung auslösen (kein sudo in Workbench)"
    },
    {
        "id": "TC-12",
        "category": "edge",
        "query": "Erkläre mir Kubernetes",
        "expected_tools": [],
        "expected_keywords": [],
        "description": "Allgemeine Frage - möglicherweise kein Tool nötig"
    },
]
