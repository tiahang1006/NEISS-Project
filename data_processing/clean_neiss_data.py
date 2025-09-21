import pandas as pd
import numpy as np
import os

# Set the working directory
os.chdir('/Users/tianchuhang/Downloads/cursor test')

def load_and_drop_columns():
    """
    Load the combined data and drop columns with too many nulls
    """
    print("Loading combined NEISS data...")
    df = pd.read_csv('neiss_combined_2016_2024.csv')
    print(f"Original shape: {df.shape}")
    
    # Drop columns with too many nulls
    columns_to_drop = ['Other_Race', 'Other_Diagnosis', 'Product_2', 'Narrative_1', 'Narrative_2']
    df_cleaned = df.drop(columns=columns_to_drop)
    print(f"After dropping high-null columns: {df_cleaned.shape}")
    print(f"Dropped columns: {columns_to_drop}")
    
    return df_cleaned

def create_body_part_mapping():
    """
    Create Body Part grouping mapping dictionary
    """
    body_part_map = {
        # 1. Head / Neck / Face
        75: "Head/Neck/Face",  # Head
        76: "Head/Neck/Face",  # Face
        77: "Head/Neck/Face",  # Eyeball
        94: "Head/Neck/Face",  # Ear
        88: "Head/Neck/Face",  # Mouth
        89: "Head/Neck/Face",  # Neck
        
        # 2. Upper Extremity
        30: "UpperExt",  # Shoulder
        32: "UpperExt",  # Elbow
        33: "UpperExt",  # Lower arm
        34: "UpperExt",  # Wrist
        80: "UpperExt",  # Upper arm
        82: "UpperExt",  # Hand
        92: "UpperExt",  # Finger
        
        # 3. Lower Extremity
        81: "LowerExt",  # Upper leg
        35: "LowerExt",  # Knee
        36: "LowerExt",  # Lower leg
        37: "LowerExt",  # Ankle
        83: "LowerExt",  # Foot
        93: "LowerExt",  # Toe
        
        # 4. Trunk
        31: "Trunk",     # Trunk upper
        79: "Trunk",     # Trunk lower
        38: "Trunk",     # Pubic region
        
        # 5. Other / Multiple / Unspecified
        0:  "Other/Multiple",  # Internal
        84: "Other/Multiple",  # 25–50% of body (burns only)
        85: "Other/Multiple",  # >50% of body
        87: "Other/Multiple"   # Not recorded
    }
    return body_part_map

def create_diagnosis_mapping():
    """
    Create Diagnosis grouping mapping dictionary
    """
    diagnosis_map = {
        # 1. Fracture
        57: "Fracture",
        
        # 2. Strain / Sprain / Dislocation
        64: "Strain/Sprain/Dislocation",
        55: "Strain/Sprain/Dislocation",
        
        # 3. Laceration / Puncture / Avulsion / Amputation
        59: "Laceration/Puncture/Avulsion/Amputation",
        63: "Laceration/Puncture/Avulsion/Amputation",
        72: "Laceration/Puncture/Avulsion/Amputation",
        50: "Laceration/Puncture/Avulsion/Amputation",
        
        # 4. Contusion / Hematoma / Internal injuries (including Hemorrhage and Concussion)
        53: "Contusion/Hematoma/Internal",
        58: "Contusion/Hematoma/Internal",
        54: "Contusion/Hematoma/Internal",
        52: "Contusion/Hematoma/Internal",
        62: "Contusion/Hematoma/Internal",
        66: "Contusion/Hematoma/Internal",  # Hemorrhage
        
        # 5. Burns (all types)
        48: "Burns",  # Scald
        51: "Burns",  # Thermal
        49: "Burns",  # Chemical
        73: "Burns",  # Radiation
        46: "Burns",  # Electrical
        47: "Burns",  # Not specified
        
        # 6. Foreign body / Aspiration / Ingestion
        41: "ForeignBody/Aspiration/Ingestion",  # Ingested
        42: "ForeignBody/Aspiration/Ingestion",  # Aspirated
        56: "ForeignBody/Aspiration/Ingestion",  # Foreign body
        
        # 7. Poisoning / Environmental / Toxic causes
        68: "Poisoning/Environmental",  # Poisoning
        65: "Poisoning/Environmental",  # Anoxia
        69: "Poisoning/Environmental",  # Submersion/Drowning
        67: "Poisoning/Environmental",  # Electric shock
        
        # 8. Other / Dental / Nerve / Skin or unspecified
        60: "Other/Dental/Nerve/Skin",  # Dental injury
        74: "Other/Dental/Nerve/Skin",  # Dermatitis/Conjunctivitis
        61: "Other/Dental/Nerve/Skin",  # Nerve damage
        71: "Other/Dental/Nerve/Skin"   # Other/Not Stated
    }
    return diagnosis_map

def clean_age_data(df):
    """
    Clean and categorize age data
    """
    print("\nCleaning age data...")
    
    # Create age categories
    age_labels = [
        "0-4",
        "5-9", 
        "10-14",
        "15-24",
        "25-44",
        "45-64",
        "65-74",
        "75-84",
        "85+"
    ]
    
    def process_age(age):
        """
        Process age values:
        - If 3-digit number starting with 2: convert to months (201-224 = 1-24 months)
        - If >= 2 and <= 120: keep as is
        - Otherwise: return NaN
        """
        if pd.isna(age):
            return np.nan
        
        age = int(age)
        
        # Handle 3-digit codes for <2 years (201-224 = 1-24 months)
        if 201 <= age <= 224:
            months = age - 200
            return months / 12.0  # Convert to years
        
        # Keep ages between 2 and 120
        elif 2 <= age <= 120:
            return age
        
        # Everything else becomes NaN
        else:
            return np.nan
    
    # Apply age processing
    df['Age_Processed'] = df['Age'].apply(process_age)
    
    # Create age categories
    def categorize_age(age):
        if pd.isna(age):
            return "Unknown"
        elif age < 5:
            return "0-4"
        elif age < 10:
            return "5-9"
        elif age < 15:
            return "10-14"
        elif age < 25:
            return "15-24"
        elif age < 45:
            return "25-44"
        elif age < 65:
            return "45-64"
        elif age < 75:
            return "65-74"
        elif age < 85:
            return "75-84"
        else:
            return "85+"
    
    df['Age_Category'] = df['Age_Processed'].apply(categorize_age)
    
    # Filter to keep only valid ages (2-120 years)
    original_count = len(df)
    df = df[df['Age_Processed'].notna()]
    filtered_count = len(df)
    
    print(f"Age filtering: {original_count} -> {filtered_count} rows (removed {original_count - filtered_count})")
    
    return df

def analyze_and_categorize_products(df):
    """
    Analyze Product_1 frequencies and create categories
    """
    print("\nAnalyzing Product_1 frequencies...")
    
    # Get top 40 products
    product_counts = df['Product_1'].value_counts()
    top_40_products = product_counts.head(40)
    
    print(f"Top 40 products by frequency:")
    for i, (product, count) in enumerate(top_40_products.items(), 1):
        print(f"{i:2d}. {product}: {count:,}")
    
    # Create product category mapping based on user-provided mapping
    product_group_map = {
        # Sports & Recreation
        1200: "Sports/Recreation",
        1205: "Sports/Recreation",  # Basketball
        1211: "Sports/Recreation",  # Football
        1233: "Sports/Recreation",  # Trampolines
        1242: "Sports/Recreation",  # Playground slide
        1244: "Sports/Recreation",  # Monkey bars
        1267: "Sports/Recreation",  # Soccer
        1333: "Sports/Recreation",  # Skateboards
        1615: "Apparel/Personal",   # Footwear
        1616: "Apparel/Personal",   # Jewelry
        1645: "Apparel/Personal",   # Day wear
        1715: "Other",              # Pet supplies
        1817: "Home/Structures",    # Porches/balconies
        1819: "Home/Tools",         # Nails/screws
        1833: "Sports/Recreation",  # (if present, cross-check)
        1842: "Home/Structures",    # Stairs/steps
        1871: "Home/Structures",    # Fences
        1884: "Home/Structures",    # Ceilings/walls
        1893: "Home/Structures",    # Doors
        1894: "Home/Structures",    # Windows
        3274: "Sports/Recreation",  # Swimming
        3265: "Sports/Recreation",  # Weight lifting
        3277: "Sports/Recreation",  # Exercise equipment
        3299: "Sports/Recreation",  # Exercise (activity/apparel)
        4014: "Home/Furniture",     # Furniture, not specified
        4056: "Home/Furniture",     # Cabinets/shelves
        4057: "Home/Furniture",     # Tables
        4074: "Home/Furniture",     # Chairs
        4076: "Home/Furniture",     # Beds
        4078: "Home/Structures",    # Ladders
        5034: "Sports/Recreation",  # Softball
        5040: "Sports/Recreation",  # Bicycles
        5041: "Sports/Recreation",  # Baseball
        604:  "Home/Furniture",     # Desks/dressers
        611:  "Home/Bathroom",      # Bathtubs/showers
        649:  "Home/Bathroom",      # Toilets
        676:  "Home/Furniture",     # Rugs/carpets
        679:  "Home/Furniture",     # Sofas
        1141: "Other",              # Containers
    }
    
    # Apply the mapping to top 40 products
    product_category_map = {}
    for product in top_40_products.index:
        if product in product_group_map:
            product_category_map[product] = product_group_map[product]
        else:
            product_category_map[product] = "Other"
    
    # Apply categorization
    def categorize_product(product):
        if product in product_category_map:
            return product_category_map[product]
        else:
            return "Other"
    
    df['Product_Category'] = df['Product_1'].apply(categorize_product)
    
    print(f"\nProduct categories created:")
    category_counts = df['Product_Category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count:,}")
    
    return df, product_category_map

def apply_mappings(df):
    """
    Apply Body_Part and Diagnosis mappings
    """
    print("\nApplying Body_Part and Diagnosis mappings...")
    
    # Apply Body_Part mapping
    body_part_map = create_body_part_mapping()
    df['Body_Part_Category'] = df['Body_Part'].map(body_part_map)
    df['Body_Part_Category'] = df['Body_Part_Category'].fillna('Unknown')
    
    # Apply Diagnosis mapping
    diagnosis_map = create_diagnosis_mapping()
    df['Diagnosis_Category'] = df['Diagnosis'].map(diagnosis_map)
    df['Diagnosis_Category'] = df['Diagnosis_Category'].fillna('Unknown')
    
    return df

def add_season_mapping(df):
    """
    Add season mapping based on Treatment_Date
    """
    print("\nAdding season mapping to Treatment_Date...")
    
    # Season mapping dictionary
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    }
    
    # Convert Treatment_Date to datetime if it's not already
    df['Treatment_Date'] = pd.to_datetime(df['Treatment_Date'])
    
    # Extract month from Treatment_Date
    df['Month'] = df['Treatment_Date'].dt.month
    
    # Apply season mapping
    df['Season'] = df['Month'].map(season_map)
    
    # Check for any missing seasons (should be none)
    missing_seasons = df['Season'].isnull().sum()
    print(f"Missing seasons: {missing_seasons}")
    
    # Show season distribution
    print(f"Season distribution:")
    season_counts = df['Season'].value_counts()
    for season, count in season_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {season}: {count:,} ({percentage:.1f}%)")
    
    return df

def generate_final_report(df):
    """
    Generate final report with category counts
    """
    print("\n" + "="*80)
    print("FINAL CLEANED DATA REPORT")
    print("="*80)
    
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    # Report for each categorical column
    categorical_columns = [
        ('Sex', 'Sex'),
        ('Race', 'Race'),
        ('Body_Part_Category', 'Body Part Category'),
        ('Diagnosis_Category', 'Diagnosis Category'),
        ('Age_Category', 'Age Category'),
        ('Product_Category', 'Product Category'),
        ('Season', 'Season'),
        ('Disposition', 'Disposition'),
        ('Location', 'Location'),
        ('Fire_Involvement', 'Fire Involvement')
    ]
    
    report_data = []
    
    for col, display_name in categorical_columns:
        if col in df.columns:
            print(f"\n{display_name}:")
            print("-" * 50)
            value_counts = df[col].value_counts()
            for value, count in value_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {value}: {count:,} ({percentage:.1f}%)")
                report_data.append({
                    'Column': display_name,
                    'Category': value,
                    'Count': count,
                    'Percentage': percentage
                })
    
    # Save detailed report to CSV
    report_df = pd.DataFrame(report_data)
    report_df.to_csv('cleaned_data_report.csv', index=False)
    print(f"\nDetailed report saved to 'cleaned_data_report.csv'")
    
    return report_df

def main():
    """
    Main data cleaning pipeline
    """
    print("Starting NEISS data cleaning pipeline...")
    
    # Step 1: Load and drop high-null columns
    df = load_and_drop_columns()
    
    # Step 2: Clean age data
    df = clean_age_data(df)
    
    # Step 3: Apply mappings
    df = apply_mappings(df)
    
    # Step 4: Add season mapping
    df = add_season_mapping(df)
    
    # Step 5: Analyze and categorize products
    df, product_map = analyze_and_categorize_products(df)
    
    # Step 6: Drop columns not needed for modeling
    columns_to_drop = [
        'CPSC_Case_Number', 'Treatment_Date', 'Age', 'Body_Part', 
        'Diagnosis', 'Product_1', 'Stratum', 'PSU', 'Weight', 
        'Source_Year', 'Age_Processed', 'Month'
    ]
    
    print(f"\nDropping columns not needed for modeling: {columns_to_drop}")
    df_final = df.drop(columns=columns_to_drop)
    print(f"Final dataset shape after dropping columns: {df_final.shape}")
    
    # Step 7: Save cleaned data
    output_file = 'neiss_cleaned_2016_2024.csv'
    df_final.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to '{output_file}'")
    
    # Step 8: Generate final report
    report_df = generate_final_report(df_final)
    
    print(f"\nData cleaning completed successfully!")
    print(f"Final dataset shape: {df.shape}")
    print(f"Output files created:")
    print(f"  - {output_file} (cleaned data)")
    print(f"  - cleaned_data_report.csv (category report)")

if __name__ == "__main__":
    main()
