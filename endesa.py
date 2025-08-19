from itertools import islice
import geopandas as gpd
import taxonomy as tx
import pandas as pd
import random
import pprint

def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def endesa():
    gp = gpd.read_file("barcelona_carrerer.gpkg")
    taxonomy = list(gp["NOM_CARRER"].unique())
    df = pd.read_excel("2022_Agrupado_Ayunt_Barcelona_Cod_Postal_08030_08033_con_Num_Calle.xlsx")

    df["street_only"] = df["STREET_TYPE__C"].astype(str) + " " + df["STREET_DESCRIPTION__C"].astype(str)
    unique_streets = df["street_only"].unique().tolist()

    df["concat"] = df["STREET_TYPE__C"].astype("str") + " " + df["STREET_DESCRIPTION__C"].astype("str") + " " + df["STREET_NUMBER__C"].astype("str")
    # streets = df["concat"].tolist()

    result = tx.apply_cities_taxonomy(unique_streets, taxonomy)

    final_results = {
        row["concat"]: result.get(row["street_only"], "__INVALID__")
        for _, row in df.iterrows()
    }

    pprint.pp(final_results)

def main():
    endesa()

if __name__ == "__main__":
    main()

