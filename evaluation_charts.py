import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter
from pathlib import Path


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


COLORS = {
    'simple': '#4CAF50',      
    'medium': '#FFC107',      
    'complex': '#F44336',     
    'edge': '#9E9E9E',        
    'primary': '#2196F3',     
    'secondary': '#673AB7',   
    'success': '#4CAF50',     
    'warning': '#FF9800',     
    'error': '#F44336'        
}


def load_results(filepath: str) -> list:
    """Lädt Evaluationsergebnisse aus JSON-Datei."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_successful(results: list) -> list:
    """Filtert nur erfolgreiche Testläufe."""
    return [r for r in results if r.get('success', False)]


def chart_latenz_kategorie(results: list, output_path: str = 'latenz_kategorie.png'):
    """
    Balkendiagramm: Durchschnittliche Latenz nach Testkategorie.
    """
    successful = filter_successful(results)
    categories = ['simple', 'medium', 'complex', 'edge']
    category_labels = ['Einfach', 'Mittel', 'Komplex', 'Edge-Cases']
    
    latencies_avg = []
    latencies_std = []
    counts = []
    
    for cat in categories:
        cat_latencies = [r['latency_seconds'] for r in successful 
                        if r.get('category') == cat]
        if cat_latencies:
            latencies_avg.append(np.mean(cat_latencies))
            latencies_std.append(np.std(cat_latencies))
            counts.append(len(cat_latencies))
        else:
            latencies_avg.append(0)
            latencies_std.append(0)
            counts.append(0)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    bars = ax.bar(category_labels, latencies_avg, 
                  color=[COLORS[cat] for cat in categories],
                  edgecolor='black', linewidth=0.8)
    
    
    ax.errorbar(category_labels, latencies_avg, yerr=latencies_std, 
                fmt='none', color='black', capsize=5)
    
    
    for bar, avg, count in zip(bars, latencies_avg, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{avg:.2f}s\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Durchschnittliche Latenz (Sekunden)')
    ax.set_xlabel('Testkategorie')
    ax.set_title('Antwortlatenz nach Komplexitätskategorie')
    ax.set_ylim(0, max(latencies_avg) * 1.3)
    
    
    overall_avg = np.mean([r['latency_seconds'] for r in successful])
    ax.axhline(y=overall_avg, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(len(categories) - 0.5, overall_avg + 0.3, f'Ø {overall_avg:.2f}s', 
            ha='right', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Gespeichert: {output_path}")


def chart_tool_nutzung(results: list, output_path: str = 'tool_nutzung.png'):
    """
    Horizontales Balkendiagramm: Häufigkeit der Tool-Nutzung.
    """
    successful = filter_successful(results)
    
    
    tool_counts = Counter()
    for r in successful:
        for tool in r.get('tools_used', []):
            tool_counts[tool] += 1
    
    
    tools = list(tool_counts.keys())
    counts = list(tool_counts.values())
    sorted_indices = np.argsort(counts)
    tools = [tools[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    
    tool_labels = []
    for t in tools:
        if len(t) > 25:
            t = t[:22] + '...'
        tool_labels.append(t)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(tool_labels, counts, color=COLORS['primary'], 
                   edgecolor='black', linewidth=0.8)
    
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{count}', ha='left', va='center', fontsize=10)
    
    ax.set_xlabel('Anzahl Aufrufe')
    ax.set_title('Häufigkeit der Tool-Nutzung')
    ax.set_xlim(0, max(counts) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Gespeichert: {output_path}")


def chart_erfolgsmetriken(results: list, output_path: str = 'erfolgsmetriken.png'):
    """
    Balkendiagramm: Erfolgsrate, Tool-Match, Keyword-Abdeckung.
    """
    total = len(results)
    successful = filter_successful(results)
    
    erfolgsrate = len(successful) / total * 100 if total > 0 else 0
    
    tool_matches = sum(1 for r in successful if r.get('tools_match', False))
    tool_match_rate = tool_matches / len(successful) * 100 if successful else 0
    
    keyword_coverages = [r.get('keywords_coverage', 0) for r in successful]
    avg_keyword_coverage = np.mean(keyword_coverages) * 100 if keyword_coverages else 0
    
    
    no_warnings = sum(1 for r in successful if r.get('num_warnings', 0) == 0)
    no_warnings_rate = no_warnings / len(successful) * 100 if successful else 0
    
    metrics = ['Erfolgsrate', 'Tool-Match', 'Keyword-\nAbdeckung', 'Ohne\nWarnungen']
    values = [erfolgsrate, tool_match_rate, avg_keyword_coverage, no_warnings_rate]
    colors = [COLORS['success'], COLORS['primary'], COLORS['secondary'], COLORS['warning']]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=0.8)
    
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Prozent (%)')
    ax.set_title('Erfolgsmetriken der Single-Agent-Evaluation')
    ax.set_ylim(0, 115)
    
   
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Gespeichert: {output_path}")


def chart_latenz_vs_tools(results: list, output_path: str = 'latenz_vs_tools.png'):
    """
    Streudiagramm: Latenz vs. Anzahl Tool-Aufrufe.
    """
    successful = filter_successful(results)
    
    tool_counts = [r.get('num_tool_calls', 0) for r in successful]
    latencies = [r.get('latency_seconds', 0) for r in successful]
    categories = [r.get('category', 'unknown') for r in successful]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
   
    for cat in ['simple', 'medium', 'complex', 'edge']:
        cat_tools = [t for t, c in zip(tool_counts, categories) if c == cat]
        cat_latencies = [l for l, c in zip(latencies, categories) if c == cat]
        ax.scatter(cat_tools, cat_latencies, c=COLORS[cat], label=cat.capitalize(),
                   s=100, edgecolors='black', linewidth=0.8, alpha=0.8)
    
    
    if len(tool_counts) > 1:
        z = np.polyfit(tool_counts, latencies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(tool_counts), max(tool_counts), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, label='Trend')
    
    ax.set_xlabel('Anzahl Tool-Aufrufe')
    ax.set_ylabel('Latenz (Sekunden)')
    ax.set_title('Korrelation: Tool-Aufrufe und Antwortzeit')
    ax.legend(loc='upper left')
    
    
    ax.set_xticks(range(0, max(tool_counts) + 2))
    ax.set_xlim(-0.5, max(tool_counts) + 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Gespeichert: {output_path}")


def chart_keyword_abdeckung(results: list, output_path: str = 'keyword_abdeckung.png'):
    """
    Balkendiagramm: Keyword-Abdeckung pro Testfall.
    """
    successful = filter_successful(results)
    
    test_ids = [r.get('test_id', f"Test {i}") for i, r in enumerate(successful)]
    coverages = [r.get('keywords_coverage', 0) * 100 for r in successful]
    categories = [r.get('category', 'unknown') for r in successful]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_colors = [COLORS[cat] for cat in categories]
    bars = ax.bar(test_ids, coverages, color=bar_colors, edgecolor='black', linewidth=0.8)
    
    
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
   
    for bar, cov in zip(bars, coverages):
        if cov < 100:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{cov:.0f}%', ha='center', va='bottom', fontsize=9, color='red')
    
    ax.set_ylabel('Keyword-Abdeckung (%)')
    ax.set_xlabel('Testfall-ID')
    ax.set_title('Keyword-Abdeckung pro Testfall')
    ax.set_ylim(0, 115)
    
    
    plt.xticks(rotation=45, ha='right')
    
    
    legend_patches = [
        mpatches.Patch(color=COLORS['simple'], label='Einfach'),
        mpatches.Patch(color=COLORS['medium'], label='Mittel'),
        mpatches.Patch(color=COLORS['complex'], label='Komplex'),
        mpatches.Patch(color=COLORS['edge'], label='Edge-Cases')
    ]
    ax.legend(handles=legend_patches, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Gespeichert: {output_path}")


def chart_zusammenfassung(results: list, output_path: str = 'zusammenfassung.png'):
    """
    Kombinierte Übersichtsgrafik (2x2 Grid).
    """
    successful = filter_successful(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    
    ax1 = axes[0, 0]
    categories = ['simple', 'medium', 'complex', 'edge']
    category_labels = ['Einfach', 'Mittel', 'Komplex', 'Edge']
    
    latencies_avg = []
    for cat in categories:
        cat_latencies = [r['latency_seconds'] for r in successful 
                        if r.get('category') == cat]
        latencies_avg.append(np.mean(cat_latencies) if cat_latencies else 0)
    
    bars1 = ax1.bar(category_labels, latencies_avg, 
                    color=[COLORS[cat] for cat in categories],
                    edgecolor='black', linewidth=0.8)
    
    for bar, avg in zip(bars1, latencies_avg):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{avg:.1f}s', ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylabel('Latenz (s)')
    ax1.set_title('Latenz nach Kategorie')
    ax1.set_ylim(0, max(latencies_avg) * 1.25)
    
   
    ax2 = axes[0, 1]
    
    tool_counts = Counter()
    for r in successful:
        for tool in r.get('tools_used', []):
           
            short_name = tool.replace('_', '\n')[:20]
            tool_counts[short_name] += 1
    
    tools = list(tool_counts.keys())[:6]  
    counts = [tool_counts[t] for t in tools]
    
    bars2 = ax2.barh(tools, counts, color=COLORS['primary'], 
                     edgecolor='black', linewidth=0.8)
    
    for bar, count in zip(bars2, counts):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{count}', ha='left', va='center', fontsize=10)
    
    ax2.set_xlabel('Anzahl')
    ax2.set_title('Tool-Nutzung')
    
    
    ax3 = axes[1, 0]
    
    total = len(results)
    erfolgsrate = len(successful) / total * 100 if total > 0 else 0
    tool_matches = sum(1 for r in successful if r.get('tools_match', False))
    tool_match_rate = tool_matches / len(successful) * 100 if successful else 0
    keyword_coverages = [r.get('keywords_coverage', 0) for r in successful]
    avg_keyword = np.mean(keyword_coverages) * 100 if keyword_coverages else 0
    
    metrics = ['Erfolg', 'Tool-Match', 'Keywords']
    values = [erfolgsrate, tool_match_rate, avg_keyword]
    colors = [COLORS['success'], COLORS['primary'], COLORS['secondary']]
    
    bars3 = ax3.bar(metrics, values, color=colors, edgecolor='black', linewidth=0.8)
    
    for bar, val in zip(bars3, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('Prozent (%)')
    ax3.set_title('Erfolgsmetriken')
    ax3.set_ylim(0, 115)
    ax3.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    
    ax4 = axes[1, 1]
    
    tool_nums = [r.get('num_tool_calls', 0) for r in successful]
    latencies = [r.get('latency_seconds', 0) for r in successful]
    cats = [r.get('category', 'unknown') for r in successful]
    
    for cat in ['simple', 'medium', 'complex', 'edge']:
        cat_tools = [t for t, c in zip(tool_nums, cats) if c == cat]
        cat_latencies = [l for l, c in zip(latencies, cats) if c == cat]
        ax4.scatter(cat_tools, cat_latencies, c=COLORS[cat], label=cat.capitalize(),
                   s=80, edgecolors='black', linewidth=0.8, alpha=0.8)
    
    ax4.set_xlabel('Tool-Aufrufe')
    ax4.set_ylabel('Latenz (s)')
    ax4.set_title('Latenz vs. Tool-Aufrufe')
    ax4.legend(loc='upper left', fontsize=9)
    
    plt.suptitle('Single-Agent Evaluation: Übersicht', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Gespeichert: {output_path}")


def print_statistics(results: list):
    """Gibt Statistiken auf der Konsole aus."""
    successful = filter_successful(results)
    
    print("\n" + "="*60)
    print("EVALUATION STATISTIKEN")
    print("="*60)
    
    print(f"\nGesamt: {len(results)} Testfälle")
    print(f"Erfolgreich: {len(successful)}")
    print(f"Fehlgeschlagen: {len(results) - len(successful)}")
    
    if successful:
        latencies = [r['latency_seconds'] for r in successful]
        print(f"\nLatenz:")
        print(f"  Minimum: {min(latencies):.2f}s")
        print(f"  Maximum: {max(latencies):.2f}s")
        print(f"  Durchschnitt: {np.mean(latencies):.2f}s")
        print(f"  Standardabweichung: {np.std(latencies):.2f}s")
        
        tool_calls = [r['num_tool_calls'] for r in successful]
        print(f"\nTool-Aufrufe:")
        print(f"  Durchschnitt: {np.mean(tool_calls):.2f}")
        print(f"  Maximum: {max(tool_calls)}")
        
        tool_matches = sum(1 for r in successful if r.get('tools_match', False))
        print(f"\nTool-Match: {tool_matches}/{len(successful)} ({tool_matches/len(successful)*100:.1f}%)")
        
        keyword_coverages = [r.get('keywords_coverage', 0) for r in successful]
        print(f"Keyword-Abdeckung: {np.mean(keyword_coverages)*100:.1f}%")
    
    print("="*60 + "\n")


def main():
    """Hauptfunktion."""
    
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
       
        eval_files = list(Path('.').glob('eval_suite*.json'))
        if eval_files:
            filepath = str(sorted(eval_files)[-1])
            print(f"Verwende: {filepath}")
        else:
            print("Keine Evaluationsdatei gefunden!")
            print("Verwendung: python evaluation_charts.py <eval_suite.json>")
            sys.exit(1)
    
    
    results = load_results(filepath)
    print(f"Geladen: {len(results)} Testfälle\n")
    
    
    print("Generiere Grafiken...")
    chart_latenz_kategorie(results)
    chart_tool_nutzung(results)
    chart_erfolgsmetriken(results)
    chart_latenz_vs_tools(results)
    chart_keyword_abdeckung(results)
    chart_zusammenfassung(results)
    
   
    print_statistics(results)
    
    print("Alle Grafiken erfolgreich generiert!")


if __name__ == "__main__":
    main()
