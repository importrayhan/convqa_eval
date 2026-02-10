"""
Data Statistics Script for SIP Format

Generates comprehensive statistics table similar to Table 1 in SIP paper:
- Number of conversations
- Number of utterances (total, user, system)
- Number of system initiatives
- Max/Avg turns per conversation
- Max/Avg actions per system turn
- Avg system initiatives per conversation
- Avg clarifying questions per conversation
- Label distribution
"""

import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_conversation(conv: Dict) -> Dict:
    """Analyze a single conversation"""
    conversations = conv.get('conversations', [])
    
    stats = {
        'num_utterances': 0,
        'num_user_utterances': 0,
        'num_system_utterances': 0,
        'num_observations': 0,
        'num_function_calls': 0,
        'num_turns': 0,
        'num_system_initiatives': 0,
        'num_clarifying_questions': 0,
        'actions_per_system_turn': [],
        'turn_labels': []
    }
    
    current_turn_actions = 0
    
    for i, utt in enumerate(conversations):
        utt_from = utt.get('from', '')
        
        if utt_from == 'human':
            stats['num_user_utterances'] += 1
            stats['num_utterances'] += 1
            current_turn_actions = 0
        
        elif utt_from == 'gpt':
            stats['num_system_utterances'] += 1
            stats['num_utterances'] += 1
            stats['num_turns'] += 1
            
            # Count actions (tools + response)
            current_turn_actions += 1
            stats['actions_per_system_turn'].append(current_turn_actions)
            
            # Check for initiative
            label = utt.get('ambiguous_type', 0)
            stats['turn_labels'].append(label)
            
            if label > 0:  # Any non-clear label is initiative
                stats['num_system_initiatives'] += 1
            
            # Check for clarifying question
            value = utt.get('value', '')
            if 'req_clarification' in value or '?' in value:
                stats['num_clarifying_questions'] += 1
        
        elif utt_from == 'observation':
            stats['num_observations'] += 1
            stats['num_utterances'] += 1
            current_turn_actions += 1
        
        elif utt_from == 'function_call':
            stats['num_function_calls'] += 1
            current_turn_actions += 1
    
    return stats


def compute_statistics(data: List[Dict], dataset_name: str = "Dataset") -> Dict:
    """Compute comprehensive statistics for dataset"""
    
    all_stats = [analyze_conversation(conv) for conv in data]
    
    # Aggregate
    stats = {
        'dataset_name': dataset_name,
        'num_conversations': len(data),
        'num_utterances': sum(s['num_utterances'] for s in all_stats),
        'num_user_utterances': sum(s['num_user_utterances'] for s in all_stats),
        'num_system_utterances': sum(s['num_system_utterances'] for s in all_stats),
        'num_observations': sum(s['num_observations'] for s in all_stats),
        'num_function_calls': sum(s['num_function_calls'] for s in all_stats),
        'num_system_initiatives': sum(s['num_system_initiatives'] for s in all_stats),
        'num_clarifying_questions': sum(s['num_clarifying_questions'] for s in all_stats),
        'max_turns_per_conv': max(s['num_turns'] for s in all_stats) if all_stats else 0,
        'avg_turns_per_conv': np.mean([s['num_turns'] for s in all_stats]) if all_stats else 0,
        'max_actions_per_system_turn': max(
            max(s['actions_per_system_turn']) if s['actions_per_system_turn'] else 0 
            for s in all_stats
        ) if all_stats else 0,
        'avg_actions_per_system_turn': np.mean([
            action 
            for s in all_stats 
            for action in s['actions_per_system_turn']
        ]) if all_stats else 0,
        'avg_system_initiatives_per_conv': np.mean([
            s['num_system_initiatives'] for s in all_stats
        ]) if all_stats else 0,
        'avg_clarifying_questions_per_conv': np.mean([
            s['num_clarifying_questions'] for s in all_stats
        ]) if all_stats else 0
    }
    
    # Label distribution
    all_labels = [label for s in all_stats for label in s['turn_labels']]
    label_counts = Counter(all_labels)
    
    stats['label_distribution'] = dict(label_counts)
    stats['label_percentages'] = {
        label: count / len(all_labels) * 100 if all_labels else 0
        for label, count in label_counts.items()
    }
    
    return stats


def create_statistics_table(stats_list: List[Dict]) -> pd.DataFrame:
    """Create Table 1 style statistics table"""
    
    rows = []
    
    for stats in stats_list:
        row = {
            'Dataset': stats['dataset_name'],
            '# conversations': stats['num_conversations'],
            '# utterances': stats['num_utterances'],
            '# user utterances': stats['num_user_utterances'],
            '# system utterances': stats['num_system_utterances'],
            '# observations': stats['num_observations'],
            '# system initiatives': stats['num_system_initiatives'],
            'Max # turns/conv.': stats['max_turns_per_conv'],
            'Avg. # turns/conv.': f"{stats['avg_turns_per_conv']:.2f}",
            'Max # actions/system turn': stats['max_actions_per_system_turn'],
            'Avg. # actions/system turn': f"{stats['avg_actions_per_system_turn']:.2f}",
            'Avg. # initiatives/conv.': f"{stats['avg_system_initiatives_per_conv']:.2f}",
            'Avg. # clarifying q/conv.': f"{stats['avg_clarifying_questions_per_conv']:.2f}"
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_label_distribution(stats_list: List[Dict], save_path: str):
    """Plot label distribution for all datasets"""
    
    class_names = {
        0: 'Clear',
        1: 'Slightly Ambiguous',
        2: 'Needs Clarification',
        3: 'Highly Ambiguous'
    }
    
    fig, axes = plt.subplots(1, len(stats_list), figsize=(6*len(stats_list), 5))
    
    if len(stats_list) == 1:
        axes = [axes]
    
    for idx, stats in enumerate(stats_list):
        dist = stats['label_percentages']
        
        labels = [class_names.get(k, f'Class {k}') for k in sorted(dist.keys())]
        values = [dist[k] for k in sorted(dist.keys())]
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b'][:len(labels)]
        
        axes[idx].bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f"{stats['dataset_name']}\nLabel Distribution", fontweight='bold')
        axes[idx].set_ylabel('Percentage (%)')
        axes[idx].set_ylim(0, 100)
        axes[idx].grid(True, axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x labels
        axes[idx].set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Label distribution plot saved: {save_path}")


def plot_conversation_length_distribution(data_list: List[List[Dict]], names: List[str], save_path: str):
    """Plot distribution of conversation lengths"""
    
    fig, axes = plt.subplots(1, len(data_list), figsize=(6*len(data_list), 5))
    
    if len(data_list) == 1:
        axes = [axes]
    
    for idx, (data, name) in enumerate(zip(data_list, names)):
        lengths = []
        for conv in data:
            num_turns = sum(1 for utt in conv.get('conversations', []) if utt.get('from') == 'gpt')
            lengths.append(num_turns)
        
        axes[idx].hist(lengths, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_title(f"{name}\nConversation Length Distribution", fontweight='bold')
        axes[idx].set_xlabel('Number of Turns')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, axis='y', alpha=0.3)
        
        # Add statistics
        axes[idx].axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lengths):.1f}')
        axes[idx].axvline(np.median(lengths), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(lengths):.1f}')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Conversation length plot saved: {save_path}")


def generate_detailed_report(stats_list: List[Dict], output_path: str):
    """Generate detailed markdown report"""
    
    with open(output_path, 'w') as f:
        f.write("# SIP Dataset Statistics Report\n\n")
        f.write(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary Statistics\n\n")
        
        for stats in stats_list:
            f.write(f"### {stats['dataset_name']}\n\n")
            f.write(f"**Basic Counts**:\n")
            f.write(f"- Total conversations: {stats['num_conversations']}\n")
            f.write(f"- Total utterances: {stats['num_utterances']}\n")
            f.write(f"  - User utterances: {stats['num_user_utterances']}\n")
            f.write(f"  - System utterances: {stats['num_system_utterances']}\n")
            f.write(f"  - Observations: {stats['num_observations']}\n")
            f.write(f"  - Function calls: {stats['num_function_calls']}\n")
            f.write(f"- System initiatives: {stats['num_system_initiatives']}\n")
            f.write(f"- Clarifying questions: {stats['num_clarifying_questions']}\n\n")
            
            f.write(f"**Conversation Statistics**:\n")
            f.write(f"- Max turns per conversation: {stats['max_turns_per_conv']}\n")
            f.write(f"- Avg turns per conversation: {stats['avg_turns_per_conv']:.2f}\n")
            f.write(f"- Max actions per system turn: {stats['max_actions_per_system_turn']}\n")
            f.write(f"- Avg actions per system turn: {stats['avg_actions_per_system_turn']:.2f}\n")
            f.write(f"- Avg initiatives per conversation: {stats['avg_system_initiatives_per_conv']:.2f}\n")
            f.write(f"- Avg clarifying questions per conversation: {stats['avg_clarifying_questions_per_conv']:.2f}\n\n")
            
            f.write(f"**Label Distribution**:\n")
            for label in sorted(stats['label_distribution'].keys()):
                count = stats['label_distribution'][label]
                pct = stats['label_percentages'][label]
                f.write(f"- Label {label}: {count} ({pct:.1f}%)\n")
            f.write("\n")
    
    print(f"  ✓ Detailed report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute SIP Dataset Statistics')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                       help='Input SIP JSON file(s)')
    parser.add_argument('--names', type=str, nargs='+',
                       help='Dataset names (default: Dataset1, Dataset2, ...)')
    parser.add_argument('--output_dir', type=str, default='data_statistics',
                       help='Output directory for statistics')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SIP Data Statistics Generator")
    print("="*70)
    print(f"Input files: {len(args.input)}")
    print("="*70 + "\n")
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    datasets = []
    dataset_names = args.names if args.names else [f"Dataset{i+1}" for i in range(len(args.input))]
    
    for filepath, name in zip(args.input, dataset_names):
        print(f"[Loading] {name} from {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        datasets.append(data)
        print(f"  Loaded {len(data)} conversations")
    
    # Compute statistics
    print("\n[Computing] Statistics...")
    stats_list = []
    for data, name in zip(datasets, dataset_names):
        stats = compute_statistics(data, name)
        stats_list.append(stats)
        print(f"  ✓ {name}")
    
    # Create table
    print("\n[Generating] Statistics table...")
    table = create_statistics_table(stats_list)
    
    print("\n" + "="*70)
    print("Table 1: Dataset Statistics")
    print("="*70)
    print(table.to_string(index=False))
    print("="*70 + "\n")
    
    # Save table
    table_path = os.path.join(args.output_dir, 'statistics_table.csv')
    table.to_csv(table_path, index=False)
    print(f"  ✓ Table saved: {table_path}")
    
    # LaTeX format
    latex_path = os.path.join(args.output_dir, 'statistics_table.tex')
    with open(latex_path, 'w') as f:
        f.write(table.to_latex(index=False))
    print(f"  ✓ LaTeX table saved: {latex_path}")
    
    # Markdown format
    markdown_path = os.path.join(args.output_dir, 'statistics_table.md')
    with open(markdown_path, 'w') as f:
        f.write(table.to_markdown(index=False))
    print(f"  ✓ Markdown table saved: {markdown_path}")
    
    # Plot label distribution
    print("\n[Plotting] Label distribution...")
    plot_path = os.path.join(args.output_dir, 'label_distribution.png')
    plot_label_distribution(stats_list, plot_path)
    
    # Plot conversation lengths
    print("[Plotting] Conversation length distribution...")
    length_path = os.path.join(args.output_dir, 'conversation_lengths.png')
    plot_conversation_length_distribution(datasets, dataset_names, length_path)
    
    # Generate detailed report
    print("[Generating] Detailed report...")
    report_path = os.path.join(args.output_dir, 'STATISTICS_REPORT.md')
    generate_detailed_report(stats_list, report_path)
    
    # Save raw statistics
    stats_json_path = os.path.join(args.output_dir, 'statistics_raw.json')
    with open(stats_json_path, 'w') as f:
        json.dump(stats_list, f, indent=2)
    print(f"  ✓ Raw statistics saved: {stats_json_path}")
    
    print("\n" + "="*70)
    print("Statistics Generation Complete!")
    print("="*70)
    print(f"All outputs saved to: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
