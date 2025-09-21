import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the cleaned NEISS data"""
    print("Loading NEISS data...")
    df = pd.read_csv('/Users/tianchuhang/Downloads/neisscode/neiss_cleaned_2016_2024.csv')
    print(f"Data loaded: {df.shape}")
    return df

def create_figure1(df):
    """Create Figure 1: Distribution of injury cases in NEISS (2016-2024)"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Figure 1. Distribution of injury cases in NEISS (2016–2024)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ========================================
    # Panel A: Body Part Category Distribution
    # ========================================
    ax1 = axes[0]
    
    # Count body part categories
    body_part_counts = df['Body_Part_Category'].value_counts()
    
    # Create bar plot
    bars = ax1.bar(range(len(body_part_counts)), body_part_counts.values, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    # Customize plot
    ax1.set_xlabel('Body Part Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Body Part Categories', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(body_part_counts)))
    ax1.set_xticklabels(body_part_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, body_part_counts.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Add percentage labels
    total_cases = body_part_counts.sum()
    for i, (bar, count) in enumerate(zip(bars, body_part_counts.values)):
        percentage = (count / total_cases) * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{percentage:.1f}%', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
    
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(body_part_counts.values) * 1.15)
    
    # ========================================
    # Panel B: Age Group Distribution
    # ========================================
    ax2 = axes[1]
    
    # Count age categories
    age_counts = df['Age_Category'].value_counts()
    
    # Define age order for better visualization
    age_order = ['0-4', '5-9', '10-14', '15-24', '25-44', '45-64', '65-74', '75-84', '85+']
    age_counts_ordered = age_counts.reindex(age_order)
    
    # Create bar plot
    bars2 = ax2.bar(range(len(age_counts_ordered)), age_counts_ordered.values,
                    color=plt.cm.viridis(np.linspace(0, 1, len(age_counts_ordered))))
    
    # Customize plot
    ax2.set_xlabel('Age Category (years)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Age Categories', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(age_counts_ordered)))
    ax2.set_xticklabels(age_counts_ordered.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars2, age_counts_ordered.values)):
        if not pd.isna(count):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000,
                    f'{count:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(age_counts_ordered.values) * 1.15)
    
    # ========================================
    # Panel C: Seasonal Distribution
    # ========================================
    ax3 = axes[2]
    
    # Count seasonal distribution
    season_counts = df['Season'].value_counts()
    
    # Define season order
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    season_counts_ordered = season_counts.reindex(season_order)
    
    # Create bar plot
    bars3 = ax3.bar(range(len(season_counts_ordered)), season_counts_ordered.values,
                     color=['#2e8b57', '#ff6347', '#daa520', '#4682b4'])
    
    # Customize plot
    ax3.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Seasonal Distribution', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xticks(range(len(season_counts_ordered)))
    ax3.set_xticklabels(season_counts_ordered.index)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars3, season_counts_ordered.values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Add percentage labels
    for i, (bar, count) in enumerate(zip(bars3, season_counts_ordered.values)):
        percentage = (count / season_counts_ordered.sum()) * 100
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{percentage:.1f}%', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
    
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, max(season_counts_ordered.values) * 1.15)
    
    # ========================================
    # Final formatting
    # ========================================
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, hspace=0.3)
    
    # Save figure
    output_path = Path('/Users/tianchuhang/Downloads/neisscode/figures')
    output_path.mkdir(exist_ok=True)
    
    plt.savefig(output_path / 'Figure1_NEISS_Distribution.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Figure 1 saved to: {output_path}")
    plt.show()
    
    return fig

def print_summary_statistics(df):
    """Print summary statistics for the figure"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS FOR FIGURE 1")
    print("="*80)
    
    # Body part distribution
    print("\n1. BODY PART DISTRIBUTION:")
    body_part_counts = df['Body_Part_Category'].value_counts()
    total_cases = body_part_counts.sum()
    for category, count in body_part_counts.items():
        percentage = (count / total_cases) * 100
        print(f"   {category:<20}: {count:>8,} ({percentage:5.1f}%)")
    
    # Age distribution
    print("\n2. AGE DISTRIBUTION:")
    age_counts = df['Age_Category'].value_counts()
    age_order = ['0-4', '5-9', '10-14', '15-24', '25-44', '45-64', '65-74', '75-84', '85+']
    for age in age_order:
        if age in age_counts:
            count = age_counts[age]
            percentage = (count / age_counts.sum()) * 100
            print(f"   {age:<10}: {count:>8,} ({percentage:5.1f}%)")
    
    # Seasonal distribution
    print("\n3. SEASONAL DISTRIBUTION:")
    season_counts = df['Season'].value_counts()
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    for season in season_order:
        if season in season_counts:
            count = season_counts[season]
            percentage = (count / season_counts.sum()) * 100
            print(f"   {season:<10}: {count:>8,} ({percentage:5.1f}%)")
    
    print(f"\nTotal cases analyzed: {len(df):,}")
    print("="*80)

def main():
    """Main function to generate Figure 1"""
    print("Generating Figure 1: NEISS Injury Distribution Analysis")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Create figure
    fig = create_figure1(df)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    print("\nFigure 1 Caption:")
    print("Figure 1. Distribution of injury cases in NEISS (2016–2024): (A) Body part categories, ")
    print("(B) Age categories, and (C) Seasonal distribution. Head/Neck/Face injuries are the most ")
    print("common, young children (0–4) and middle-aged adults (25–44) have high counts, and ")
    print("incidence peaks in summer months.")

if __name__ == "__main__":
    main()
