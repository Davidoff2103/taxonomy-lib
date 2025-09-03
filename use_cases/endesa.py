import geopandas as gpd
import os
import sys

# Add parent directory to Python path to access the taxonomy module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import taxonomy as tx
import pandas as pd
import pprint

def endesa():
    """
    Process Endesa's address data to match against Barcelona street taxonomy.
    
    This function:
    1. Loads Barcelona street names from a GeoPackage file
    2. Reads Endesa's address data from an Excel file
    3. Applies a taxonomy similarity matching algorithm to identify valid addresses
    4. Outputs a dictionary mapping full addresses to their taxonomy matches
    
    The workflow uses a combination of street type, description, and number fields
    to perform address validation against the official street names list.
    """
    # Load Barcelona street names from GeoPackage file
    # Path: ../barcelona_carrerer.gpkg (relative to this script)
    gp = gpd.read_file("../barcelona_carrerer.gpkg")
    
    # Create taxonomy list from unique street names in the geospatial dataset
    taxonomy = list(gp["NOM_CARRER"].unique())
    
    # Read Endesa's address data from Excel file
    # Path: ../Agrupo_Ayunt_Barcelona_Cod_Postal_08002_08003-1.xlsx
    # (Original file path is commented out as an alternative reference)
    # df = pd.read_excel("2022_Agrupado_Ayunt_Barcelona_Cod_Postal_08030_08033_con_Num_Calle.xlsx")
    df = pd.read_excel("../Agrupo_Ayunt_Barcelona_Cod_Postal_08002_08003-1.xlsx")
    
    # Create a 'street_only' column by combining street type and description
    # Format: "TYPE DESCRIPTION" (e.g., "Carrer de la Rambla")
    df["street_only"] = (df["STREET_TYPE__C"].astype(str) + " " +
                         df["STREET_DESCRIPTION__C"].astype(str))
    
    # Extract list of unique street names from Endesa data
    unique_streets = df["street_only"].unique().tolist()
    
    # Create a 'concat' column combining type, description, and number
    # Format: "TYPE DESCRIPTION NUMBER" (e.g., "Carrer de la Rambla 123")
    df["concat"] = (df["STREET_TYPE__C"].astype("str") + " " +
                     df["STREET_DESCRIPTION__C"].astype("str") + " " +
                     df["STREET_NUMBER__C"].astype("str"))
    
    # Apply taxonomy similarity matching to find best street name matches
    # Returns a dictionary: {street_name: taxonomy_match or "__INVALID__"}
    result = tx.apply_taxonomy_similarity(unique_streets, taxonomy, "streets")
    
    # Build final results dictionary mapping full addresses to their taxonomy matches
    # Uses the 'concat' field as the key and looks up the result for the 'street_only' part
    final_results = {
        row["concat"]: result.get(row["street_only"], "__INVALID__")
        for _, row in df.iterrows()
    }
    
    # Pretty-print the final results dictionary
    pprint.pp(final_results)

def main():
    """
    Entry point for the Endesa use case.
    
    Simply calls the endesa() function to execute the address matching workflow.
    """
    endesa()

if __name__ == "__main__":
    main()