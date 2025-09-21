import pandas as pd
import numpy as np

def load_and_prepare_data():
    """
    Load the cleaned data and prepare it for Table 1 generation
    """
    print("Loading cleaned NEISS data...")
    df = pd.read_csv('neiss_cleaned_2016_2024.csv')
    print(f"Data loaded: {df.shape}")
    return df

def replace_codes_with_names(df):
    """
    Replace all codes with human-readable names
    """
    print("Replacing codes with human-readable names...")
    
    # Sex mapping
    sex_mapping = {
        1.0: "Male",
        2.0: "Female", 
        3.0: "Gender diverse/Intersex",
        0.0: "Missing/Unknown"
    }
    df['Sex'] = df['Sex'].map(sex_mapping)
    
    # Race mapping
    race_mapping = {
        1: "White",
        2: "Black/African American",
        3: "Other",
        4: "Asian",
        5: "American Indian/Alaska Native",
        6: "Native Hawaiian/Pacific Islander",
        0: "Missing/Unknown"
    }
    df['Race'] = df['Race'].map(race_mapping)
    
    # Disposition mapping
    disposition_mapping = {
        1: "Treated and released",
        2: "Treated & transferred",
        4: "Admitted",
        5: "Observation",
        6: "Left/AMA/Eloped",
        8: "Fatality",
        9: "Missing/Unknown"
    }
    df['Disposition'] = df['Disposition'].map(disposition_mapping)
    
    # Location mapping
    location_mapping = {
        0: "Missing/Unknown",
        1: "Home",
        2: "Farm/Ranch",
        4: "Street/Highway",
        5: "Other public property",
        6: "Mobile home",
        7: "Industrial place",
        8: "School",
        9: "Recreation/Sports place"
    }
    df['Location'] = df['Location'].map(location_mapping)
    
    # Fire Involvement mapping
    fire_mapping = {
        0: "No fire involvement",
        1: "Fire involvement",
        2: "Fire involvement", 
        3: "Fire involvement"
    }
    df['Fire_Involvement'] = df['Fire_Involvement'].map(fire_mapping)
    
    # Merge Unknown/Not stated/Not recorded into Missing/Unknown for all variables
    for col in ['Age_Category', 'Diagnosis_Category', 'Product_Category', 'Season']:
        df[col] = df[col].replace(['Unknown', 'Not stated', 'Not recorded'], 'Missing/Unknown')
    
    return df

def generate_table1(df):
    """
    Generate Table 1 in journal style
    """
    print("Generating Table 1...")
    
    # Define the variables to include in Table 1
    variables = ['Sex', 'Race', 'Age_Category', 'Diagnosis_Category', 
                'Product_Category', 'Season', 'Location', 'Disposition', 'Fire_Involvement']
    
    # Get body part categories
    body_part_categories = df['Body_Part_Category'].value_counts().index.tolist()
    
    # Initialize results
    results = []
    
    # Calculate overall totals
    total_n = len(df)
    
    for variable in variables:
        print(f"Processing {variable}...")
        
        # Get value counts for overall
        overall_counts = df[variable].value_counts()
        
        # Add variable name as main row
        row_data = {'Characteristic': variable, 'Overall': f"{total_n:,} (100.0%)"}
        for bp_cat in body_part_categories:
            row_data[f'{bp_cat}'] = ""
        results.append(row_data)
        
        # Add subcategories
        for value in overall_counts.index:
            n_overall = overall_counts[value]
            pct_overall = (n_overall / total_n) * 100
            
            row_data = {
                'Characteristic': f"  {value}",
                'Overall': f"{n_overall:,} ({pct_overall:.1f}%)"
            }
            
            # Calculate for each body part category
            for bp_cat in body_part_categories:
                bp_data = df[df['Body_Part_Category'] == bp_cat]
                bp_total = len(bp_data)
                
                if bp_total > 0:
                    bp_count = len(bp_data[bp_data[variable] == value])
                    bp_pct = (bp_count / bp_total) * 100
                    row_data[f'{bp_cat}'] = f"{bp_count:,} ({bp_pct:.1f}%)"
                else:
                    row_data[f'{bp_cat}'] = "0 (0.0%)"
            
            results.append(row_data)
    
    # Convert to DataFrame
    table1_df = pd.DataFrame(results)
    
    return table1_df, body_part_categories

def format_table1_for_publication(table1_df, body_part_categories):
    """
    Format Table 1 for publication
    """
    print("Formatting Table 1 for publication...")
    
    # Create the formatted table
    print("\n" + "="*120)
    print("Table 1. Baseline Characteristics of the Study Population by Body Part Category (NEISS 2016–2024)")
    print("="*120)
    
    # Header
    header = f"{'Characteristic':<30} {'Overall n (%)':<15}"
    for bp_cat in body_part_categories:
        header += f" {bp_cat} n (%)".ljust(20)
    print(header)
    print("-" * 120)
    
    # Data rows
    for _, row in table1_df.iterrows():
        line = f"{row['Characteristic']:<30} {row['Overall']:<15}"
        for bp_cat in body_part_categories:
            line += f" {row[bp_cat]:<19}"
        print(line)
    
    print("="*120)
    
    return table1_df

def save_table1_to_csv(table1_df, body_part_categories):
    """
    Save Table 1 to CSV file
    """
    # Rename columns for better readability
    column_names = ['Characteristic', 'Overall'] + body_part_categories
    table1_df.columns = column_names
    
    # Save to CSV
    output_file = 'table1_baseline_characteristics.csv'
    table1_df.to_csv(output_file, index=False)
    print(f"\nTable 1 saved to: {output_file}")
    
    return output_file

def main():
    """
    Main function to generate Table 1
    """
    print("Generating Table 1: Baseline Characteristics of the Study Population")
    print("="*80)
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data()
    
    # Step 2: Replace codes with human-readable names
    df = replace_codes_with_names(df)
    
    # Step 3: Generate Table 1
    table1_df, body_part_categories = generate_table1(df)
    
    # Step 4: Format and display Table 1
    table1_df = format_table1_for_publication(table1_df, body_part_categories)
    
    # Step 5: Save to CSV
    output_file = save_table1_to_csv(table1_df, body_part_categories)
    
    print(f"\nTable 1 generation completed successfully!")
    print(f"Output file: {output_file}")
    print(f"Total rows in dataset: {len(df):,}")

if __name__ == "__main__":
    main()
