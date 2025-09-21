import pandas as pd
import os
import glob
from pathlib import Path

# Set the working directory
os.chdir('/Users/tianchuhang/Downloads/neisscode')

def concatenate_neiss_data():
    """
    Concatenate all NEISS Excel files (2016-2024) using only the columns from neiss2016.xlsx
    """
    print("Starting NEISS data concatenation...")
    
    # First, read neiss2016.xlsx to get the base columns
    print("Reading neiss2016.xlsx to get base columns...")
    try:
        df_2016 = pd.read_excel('neiss_data/neiss2016.xlsx', sheet_name=0)
        base_columns = list(df_2016.columns)
        print(f"Base columns from 2016: {base_columns}")
        print(f"Number of columns: {len(base_columns)}")
        
        # Initialize the combined dataframe with 2016 data
        combined_df = df_2016.copy()
        print(f"neiss2016.xlsx loaded: {df_2016.shape[0]} rows")
        
    except Exception as e:
        print(f"Error reading neiss2016.xlsx: {e}")
        return None
    
    # Get all NEISS files from 2017 to 2024
    neiss_files = []
    for year in range(2017, 2025):
        filename = f'neiss_data/neiss{year}.xlsx'
        if os.path.exists(filename):
            neiss_files.append(filename)
        else:
            print(f"Warning: {filename} not found")
    
    print(f"Found {len(neiss_files)} additional files: {neiss_files}")
    
    # Process each file
    for filename in neiss_files:
        print(f"\nProcessing {filename}...")
        try:
            # Read the Excel file
            df = pd.read_excel(filename, sheet_name=0)
            print(f"  Original shape: {df.shape}")
            
            # Check which columns are available
            available_columns = [col for col in base_columns if col in df.columns]
            missing_columns = [col for col in base_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  Missing columns: {missing_columns}")
                # Add missing columns with NaN values
                for col in missing_columns:
                    df[col] = pd.NA
            
            # Select only the base columns (this will drop extra columns)
            df_filtered = df[base_columns]
            print(f"  After filtering to base columns: {df_filtered.shape}")
            
            # Add a year column to track the source
            df_filtered['Source_Year'] = filename.replace('neiss', '').replace('.xlsx', '')
            
            # Concatenate to the combined dataframe
            combined_df = pd.concat([combined_df, df_filtered], ignore_index=True)
            print(f"  Added {df_filtered.shape[0]} rows. Total rows: {combined_df.shape[0]}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    
    print(f"\nFinal combined dataset shape: {combined_df.shape}")
    return combined_df

def analyze_null_values(df):
    """
    Analyze and display null/NA values for each column
    """
    print("\n" + "="*60)
    print("NULL/NA VALUES ANALYSIS")
    print("="*60)
    
    total_rows = len(df)
    print(f"Total number of rows: {total_rows:,}")
    print("\nNull/NA values by column:")
    print("-" * 60)
    
    null_analysis = []
    for column in df.columns:
        if column == 'Source_Year':  # Skip the source year column for null analysis
            continue
            
        null_count = df[column].isnull().sum()
        null_percentage = (null_count / total_rows) * 100
        
        null_analysis.append({
            'Column': column,
            'Null_Count': null_count,
            'Null_Percentage': null_percentage,
            'Non_Null_Count': total_rows - null_count
        })
        
        print(f"{column:25} | {null_count:8,} ({null_percentage:5.1f}%) | {total_rows - null_count:8,} non-null")
    
    return null_analysis

def main():
    # Concatenate all data
    combined_df = concatenate_neiss_data()
    
    if combined_df is not None:
        # Save to CSV
        output_filename = 'neiss_combined_2016_2024.csv'
        print(f"\nSaving combined data to {output_filename}...")
        combined_df.to_csv(output_filename, index=False)
        print(f"Data saved successfully to {output_filename}")
        
        # Analyze null values
        null_analysis = analyze_null_values(combined_df)
        
        # Save null analysis to a text file
        with open('null_analysis_report.txt', 'w') as f:
            f.write("NEISS Data Concatenation Report\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total rows: {len(combined_df):,}\n")
            f.write(f"Total columns: {len(combined_df.columns)}\n")
            f.write(f"Output file: {output_filename}\n\n")
            f.write("Null/NA Values Analysis:\n")
            f.write("-" * 30 + "\n")
            for analysis in null_analysis:
                f.write(f"{analysis['Column']:25} | {analysis['Null_Count']:8,} ({analysis['Null_Percentage']:5.1f}%)\n")
        
        print(f"\nNull analysis report saved to null_analysis_report.txt")
        print(f"Combined data saved to {output_filename}")
        
    else:
        print("Failed to concatenate data")

if __name__ == "__main__":
    main()
