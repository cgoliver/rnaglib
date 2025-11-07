#!/usr/bin/env python3
"""
Generate a modern, interactive count plot showing the number of RNAs in each task dataset.
Exports to SVG for embedding in documentation.
"""

import os
import sys
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except ImportError:
    print("Warning: plotly not installed. Install with: pip install plotly kaleido")
    sys.exit(1)

# Data from the dataframe
all_data = {
    'task_id': ['rna_cm', 'rna_go', 'rna_if', 'rna_if_bench', 'rna_ligand', 'rna_prot', 'rna_site', 'rna_site_bench'],
    'size': [357, 499, 10154, 11946, 488, 3187, 634, 76]
}

# Task display names
task_display_names = {
    'rna_cm': 'Chemical<br>Modification',
    'rna_go': 'RNA GO',
    'rna_if': 'Inverse<br>Folding',
    'rna_if_bench': 'gRNAde',
    'rna_ligand': 'Ligand<br>Identification',
    'rna_prot': 'Protein<br>Binding Site',
    'rna_site': 'Binding Site',
    'rna_site_bench': 'Benchmark<br>Binding Site'
}

# Split data into two groups
inverse_folding_tasks = ['rna_if', 'rna_if_bench']
other_tasks = ['rna_cm', 'rna_go', 'rna_ligand', 'rna_prot', 'rna_site', 'rna_site_bench']

def generate_single_plot(task_ids, sizes, labels, title, output_path):
    """Generate a single modern, interactive count plot.
    
    :param task_ids: List of task IDs
    :param sizes: List of dataset sizes
    :param labels: List of display labels
    :param title: Plot title
    :param output_path: Path where the plot will be saved (SVG format)
    """
    
    # Create interactive bar chart with modern styling
    fig = go.Figure()
    
    # Add bars with single purple color
    fig.add_trace(go.Bar(
        x=labels,
        y=sizes,
        marker=dict(
            color='#e9d5ff',  # Light purple fill
            line=dict(color='#6b21a8', width=2)  # Dark purple outline
        ),
        text=[f'{size:,}' for size in sizes],
        textposition='outside',
        textfont=dict(size=11, family='Arial, sans-serif', color='#6b21a8', weight='bold'),
        hovertemplate='<b>%{x}</b><br>' +
                      'Dataset Size: <b>%{y:,}</b> RNAs<br>' +
                      '<extra></extra>',
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='#9333ea',
            font_size=12,
            font_family='Arial, sans-serif'
        )
    ))
    
    # Update layout with modern styling
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=18, family='Arial, sans-serif', color='#1f2937'),
            pad=dict(t=20, b=30)
        ),
        xaxis=dict(
            title='',
            tickfont=dict(size=11, family='Arial, sans-serif', color='#6b7280'),
            gridcolor='rgba(147, 51, 234, 0.1)',
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text='Number of RNAs', font=dict(size=14, family='Arial, sans-serif', color='#4b5563')),
            tickfont=dict(size=11, family='Arial, sans-serif', color='#6b7280'),
            gridcolor='rgba(147, 51, 234, 0.1)',
            showgrid=True,
            gridwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=20, r=20, t=80, b=60),
        hovermode='closest',
        font=dict(family='Arial, sans-serif')
    )
    
    # Save as SVG
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try to save as SVG using kaleido
        pio.write_image(fig, str(output_path), format='svg', width=1200, height=500, scale=2)
        print(f"SVG plot saved to: {output_path}")
        
        # Also save as HTML for interactive version
        html_path = output_path.with_suffix('.html')
        fig.write_html(str(html_path))
        print(f"Interactive HTML plot saved to: {html_path}")
        
    except Exception as e:
        print(f"Warning: Could not save as SVG (kaleido may not be installed): {e}")
        print("Saving as HTML instead (interactive but not SVG format)")
        html_path = output_path.with_suffix('.html')
        fig.write_html(str(html_path))
        print(f"Interactive HTML plot saved to: {html_path}")

def generate_side_by_side_plot(base_output_path):
    """Generate a single plot with two subplots side by side.
    
    :param base_output_path: Base path for output files (without extension)
    """
    # Prepare inverse folding tasks data
    if_task_ids = [tid for tid in all_data['task_id'] if tid in inverse_folding_tasks]
    if_sizes = [all_data['size'][i] for i, tid in enumerate(all_data['task_id']) if tid in inverse_folding_tasks]
    if_labels = [task_display_names.get(tid, tid) for tid in if_task_ids]
    
    # Sort inverse folding tasks by size (low to high)
    if_sorted = sorted(zip(if_task_ids, if_sizes, if_labels), key=lambda x: x[1])
    if_task_ids, if_sizes, if_labels = zip(*if_sorted) if if_sorted else ([], [], [])
    if_task_ids, if_sizes, if_labels = list(if_task_ids), list(if_sizes), list(if_labels)
    
    # Prepare other tasks data
    other_task_ids = [tid for tid in all_data['task_id'] if tid in other_tasks]
    other_sizes = [all_data['size'][i] for i, tid in enumerate(all_data['task_id']) if tid in other_tasks]
    other_labels = [task_display_names.get(tid, tid) for tid in other_task_ids]
    
    # Sort other tasks by size (low to high)
    other_sorted = sorted(zip(other_task_ids, other_sizes, other_labels), key=lambda x: x[1])
    other_task_ids, other_sizes, other_labels = zip(*other_sorted) if other_sorted else ([], [], [])
    other_task_ids, other_sizes, other_labels = list(other_task_ids), list(other_sizes), list(other_labels)
    
    # Create subplots side by side (swapped order: other tasks first, then inverse folding)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('', ''),  # No subplot titles
        horizontal_spacing=0.05  # Reduced spacing
    )
    
    # Calculate consistent bar width based on the number of bars
    # Use the maximum number of bars to determine a base width
    max_bars = max(len(other_labels), len(if_labels))
    # Set a consistent bar width that works for both plots
    bar_width = 0.6  # Fixed width that will be consistent across both subplots
    
    # Add other tasks bars to left subplot (swapped)
    fig.add_trace(
        go.Bar(
            x=other_labels,
            y=other_sizes,
            width=bar_width,
            marker=dict(
                color='#e9d5ff',  # Light purple fill
                line=dict(color='#6b21a8', width=2)  # Dark purple outline
            ),
            text=[f'{size:,}' for size in other_sizes],
            textposition='outside',
            textfont=dict(size=11, family='Arial, sans-serif', color='#6b21a8', weight='bold'),
            hovertemplate='<b>%{x}</b><br>Dataset Size: <b>%{y:,}</b> RNAs<extra></extra>',
            hoverlabel=dict(
                bgcolor='white',
                bordercolor='#9333ea',
                font_size=12,
                font_family='Arial, sans-serif'
            ),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add inverse folding bars to right subplot (swapped)
    fig.add_trace(
        go.Bar(
            x=if_labels,
            y=if_sizes,
            width=bar_width,
            marker=dict(
                color='#e9d5ff',  # Light purple fill
                line=dict(color='#6b21a8', width=2)  # Dark purple outline
            ),
            text=[f'{size:,}' for size in if_sizes],
            textposition='outside',
            textfont=dict(size=11, family='Arial, sans-serif', color='#6b21a8', weight='bold'),
            hovertemplate='<b>%{x}</b><br>Dataset Size: <b>%{y:,}</b> RNAs<extra></extra>',
            hoverlabel=dict(
                bgcolor='white',
                bordercolor='#9333ea',
                font_size=12,
                font_family='Arial, sans-serif'
            ),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update layout (no title)
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=20, r=20, t=60, b=60),
        hovermode='closest',
        font=dict(family='Arial, sans-serif')
    )
    
    # Update x-axes
    fig.update_xaxes(
        title='',
        tickfont=dict(size=11, family='Arial, sans-serif', color='#6b7280'),
        showgrid=False,
        row=1, col=1
    )
    fig.update_xaxes(
        title='',
        tickfont=dict(size=11, family='Arial, sans-serif', color='#6b7280'),
        showgrid=False,
        row=1, col=2
    )
    
    # Update y-axes
    fig.update_yaxes(
        title=dict(text='Number of RNAs', font=dict(size=14, family='Arial, sans-serif', color='#4b5563')),
        tickfont=dict(size=11, family='Arial, sans-serif', color='#6b7280'),
        gridcolor='rgba(147, 51, 234, 0.1)',
        showgrid=True,
        gridwidth=1,
        row=1, col=1
    )
    fig.update_yaxes(
        title=dict(text='Number of RNAs', font=dict(size=14, family='Arial, sans-serif', color='#4b5563')),
        tickfont=dict(size=11, family='Arial, sans-serif', color='#6b7280'),
        gridcolor='rgba(147, 51, 234, 0.1)',
        showgrid=True,
        gridwidth=1,
        row=1, col=2
    )
    
    # Remove subplot title annotations (already set to empty strings)
    
    # Save as SVG
    output_path = Path(base_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try to save as SVG using kaleido
        pio.write_image(fig, str(output_path), format='svg', width=1600, height=500, scale=2)
        print(f"SVG plot saved to: {output_path}")
        
        # Also save as HTML for interactive version
        html_path = output_path.with_suffix('.html')
        fig.write_html(str(html_path))
        print(f"Interactive HTML plot saved to: {html_path}")
        
    except Exception as e:
        print(f"Warning: Could not save as SVG (kaleido may not be installed): {e}")
        print("Saving as HTML instead (interactive but not SVG format)")
        html_path = output_path.with_suffix('.html')
        fig.write_html(str(html_path))
        print(f"Interactive HTML plot saved to: {html_path}")

def generate_task_plots(base_output_path):
    """Generate two separate plots: one for inverse folding tasks, one for others.
    
    :param base_output_path: Base path for output files (without extension)
    """
    # Use the side-by-side plot function
    generate_side_by_side_plot(base_output_path)

if __name__ == "__main__":
    # Get output path from command line or use default
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        # Default to images directory in source
        output_path = os.path.join(os.path.dirname(__file__), 'images', 'task_dataset_sizes.svg')
    
    generate_task_plots(output_path)

