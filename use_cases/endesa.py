import geopandas as gpd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import taxonomy as tx
import pandas as pd
import pprint


def endesa():
    gp = gpd.read_file("../barcelona_carrerer.gpkg")
    taxonomy = list(gp["NOM_CARRER"].unique())
#    df = pd.read_excel("2022_Agrupado_Ayunt_Barcelona_Cod_Postal_08030_08033_con_Num_Calle.xlsx")
    df = pd.read_excel("../Agrupo_Ayunt_Barcelona_Cod_Postal_08002_08003-1.xlsx")
    df["street_only"] = (df["STREET_TYPE__C"].astype(str) + " " +
                         df["STREET_DESCRIPTION__C"].astype(str))
    unique_streets = df["street_only"].unique().tolist()

    df["concat"] = (df["STREET_TYPE__C"].astype("str") + " " +
                    df["STREET_DESCRIPTION__C"].astype("str") + " " +
                    df["STREET_NUMBER__C"].astype("str"))
    # streets = df["concat"].tolist()

    result = tx.apply_taxonomy_similarity(unique_streets, taxonomy, "streets")

    final_results = {
        row["concat"]: result.get(row["street_only"], "__INVALID__")
        for _, row in df.iterrows()
    }

    pprint.pp(final_results)


def main():
    endesa()


if __name__ == "__main__":
    main()
