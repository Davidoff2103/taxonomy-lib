import os
import sys

# Add parent directory to Python path to import the taxonomy module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import taxonomy as tx
import pprint

def calderes():
    """
    Process building envelope type taxonomy matching for the Calderes use case.
    
    This function analyzes a list of building envelope-related terms against 
    predefined taxonomies to identify the most appropriate classification for each
    term using the taxonomy library's reasoning function.
    
    Returns:
        dict: A dictionary mapping input terms to their taxonomy classifications
    """
    # List of terms related to building envelope components and thermal bridges
    values = [
        'Puente térmico encuentro de fachadas en esquina saliente', 
        'Cubierta', 
        'Trobada de façana amb coberta', 
        'Alfeizar', 
        'Encuentro de fachada con solera', 
        'Trobada de façana amb forjat', 
        'ParticionInteriorHorizontal', 
        'Puente térmico lineal Jamba en huecos', 
        'Puente térmico lineal Alfeizar en huecos', 
        'Cantonada entrant de fa�anas', 
        'Jamba', 
        'Encuentro de fachada con cubierta', 
        'Puente térmico lineal pilares integrados en fachadas', 
        'Suelo', 
        'Muro Exterior', 
        'ESQUINA_CONVEXA_FORJADO', 
        'Pilar integrado en fachada', 
        'Trobada de fa�ana amb solera', 
        'Puente térmico encuentro de fachada con solera', 
        'Frente de forjado', 
        'ESQUINA_CONVEXA_CERRAMIENTO', 
        'Contorno de hueco', 
        "Trobada de faC'ana amb coberta", 
        'Esquina', 
        'Separación No Habitable', 
        'Dintel', 
        'Fachada', 
        'Trobada de façana amb voladís', 
        'UNION_SOLERA_PAREDEXT', 
        'PILLAR', 
        'Puente térmico lineal Dintel en huecos', 
        'Cantonada sortint de fa�anes', 
        'ESQUINA_CONCAVA_CERRAMIENTO', 
        'UNION_CUBIERTA', 
        'Puente térmico lineal Capialzado en huecos', 
        'Adiabatico', 
        'Hueco', 
        'Puente térmico encuentro de fachada con suelo en contacto con el aire', 
        'Pilar', 
        'Encuentro de fachada con suelo en contacto con el aire', 
        'Muro Contacto Terreno', 
        'Cantonada entrant de façanes', 
        "Trobada de faC'ana amb forjat", 
        'Trobada de fa�ana amb coberta', 
        'Trobada de fa�ana amb forjat', 
        'Puente térmico encuentro de fachada con cubierta', 
        'FRENTE_FORJADO', 
        'ParticionInteriorVertical', 
        "Cantonada entrant de faC'anes", 
        'Pilar en Esquina', 
        'Esquina entrante de fachadas', 
        'Trobada de fa�ana amb volad�s', 
        'Lucernario', 
        'Trobada de façana amb solera', 
        "Trobada de faC'ana amb solera", 
        'Encuentro de fachada con forjado', 
        'Esquina hacia el exterior', 
        'Buit de finestra', 
        'Cantonada sortint de façanes', 
        "Cantonada entrant de faC'anes", 
        'Esquina hacia el interior', 
        'Esquina saliente de fachadas', 
        'Puente térmico encuentro de fachadas en esquina entrante', 
        'Encuentro de fachada con voladizo'
    ]

    # List of predefined taxonomies for building envelope classification
    taxonomies = [
        'NonHabitableAreaContactEnvelope', 
        'Window', 
        'TerrainContactEnvelope', 
        'AirContactEnvelope', 
        'ThermalBridge', 
        'AirContactRoof', 
        'IndoorEnvelope'
    ]

    # Apply taxonomy reasoning to map input terms to their appropriate taxonomy
    # categories using the taxonomy library's classification function
    result = tx.apply_taxonomy_reasoning(values, taxonomies, "Building envelope type")

    # Pretty-print the classification results
    pprint.pp(result)

def main():
    """Entry point for the Calderes use case execution."""
    calderes()

if __name__ == "__main__":
    main()