import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import taxonomy as tx
import pprint


def translate():
    headers = ['Personas de 16 años o más - Lugar de trabajo/estudio',
               'Personas de 16 años o más - Número de desplazamaientos',
               'Personas de 16 años o más - Medio de transporte',
               'Personas de 16 años o más - Tipo de vehículo',
               'Personas de 16 años o más - Tiempo diario',
               'Personas de 16 años o más - Grado de satisfacción',
               'Personas de 16 años o más - Grado de participación en las tareas domésticas',
               'Personas de 16 años o más que conviven con otras personas - Grado de participación en cuidados a menores o dependientes dentro del hogar',
               'Personas de 16 años o más que conviven con otras personas y participan en cuidados a personas dependientes dentro del hogar - Tipo de dependiente',
               'Personas de 16 años o más que conviven con otras personas y participan en cuidados a personas dependientes dentro del hogar - Horas diarias dedicadas al cuidado',
               'Personas de 16 años o más - Grado de participación en cuidados a menores o dependientes fuera del hogar',
               'Personas de 16 años o más que participan en cuidados a personas dependientes fuera del hogar - Tipo de dependiente',
               'Personas de 16 años o más que participan en cuidados a personas dependientes fuera del hogar - Horas diarias dedicadas al cuidado',
               'Personas de 16 años o más - Tiene apoyo social',
               'Personas de 16 años o más que viven solas - Tipo de relación',
               'Personas de 16 años o más que viven solas - Lugar de residencia',
               'Personas nacidas en España - Lugar de nacimiento progenitores',
               'Personas nacidas fuera de España - Lugar de nacimiento progenitores',
               'Personas - Residencia progenitores',
               'Personas - Nacionalidad progenitores',
               'Personas de 16 años o más - Nivel estudios progenitores',
               'Personas de 16 años o más - Accede habitualmente a internet',
               'Personas de 16 años o más - Dispone de perfil en alguna red social',
               'Personas de 16 años o más - Dispone de smartphone',
               'Personas de 16 años o más - Realiza compras por internet',
               'Personas de 16 años o más - Realiza ventas por internet',
               'Hogares de una sola persona (unipersonales) - Edad',
               'Hogares de una sola persona (unipersonales) - Nacionalidad',
               'Hogares de una sola persona (unipersonales) - Estado civil',
               'Hogares de una sola persona (unipersonales) - Nivel educativo',
               'Hogares de una sola persona (unipersonales) - Situación laboral',
               'Hogares de una sola persona (unipersonales) - Nivel de ingresos mensuales netos del hogar',
               'Parejas convivientes - Sexo de la pareja',
               'Parejas convivientes - Nacionalidad de la pareja',
               'Parejas convivientes - Número de hijos',
               'Hogares - Número de hijos',
               'Hogares - Número de habitaciones de la vivienda',
               'Hogares - Superficie útil de la vivienda',
               'Hogares - Tipo de edificio',
               'Hogares - Régimen de tenencia de la vivienda',
               'Hogares/personas - Régimen de tenencia de la vivienda',
               'Hogares/personas en viviendas alquiladas - Cuota mensual del alquiler',
               'Hogares/personas en viviendas propias, - Cuota mensual de la hipoteca',
               'Hogares/personas - Disponen de segunda residencia',
               'Hogares - Disponen de segunda residencia',
               'Hogares/personas con segunda residencia - Lugar de la segunda residencia',
               'Hogares/personas con segunda residencia - Días de uso de la segunda residencia al año',
               'Hogares/personas - Número de vehículos',
               'Hogares - Número de vehículos',
               'Hogares/personas con vehículo - Vehículo ecólogico',
               'Hogares con vehículo - Vehículo ecólogico',
               'Hogares/personas - Separan algún tipo de residuo',
               'Hogares - Separan algún tipo de residuo',
               'Hogares - Tipo de residuos separados',
               'Hogares/personas con servicio doméstico remunerado - Servicio doméstico remunerado',
               'Hogares con servicio doméstico remunerado - Servicio doméstico remunerado',
               'Hogares/personas con ayuda externa - Ayuda externa',
               'Hogares con ayuda externa - Ayuda externa',
               'Viviendas principales - Adaptada a necesidades propias del envejecimiento',
               'Viviendas principales - Problema de aislamiento',
               'Viviendas principales - Tipo de calefacción',
               'Viviendas principales con calefacción - Tipo de combustible',
               'Viviendas principales - Sistema de suministro de agua',
               'Viviendas principales - Tiene sistema de refrigeración',
               'Viviendas principales - Tipo de conexión a internet',
               'Viviendas principales - Tipo de electrodoméstico',
               'Viviendas principales - Tipo de bombillas',
               'Viviendas principales - Superficie útil de la vivienda',
               'Viviendas principales - Aseo con inodoro / Bañera o ducha',
               'Viviendas principales - Número de cuartos de baño o aseos',
               'Viviendas principales - Cocina independiente de 4 m2 o más',
               'Viviendas principales - Número de habitaciones',
               'Viviendas principales - Tipo de problemática en la zona',
               'Viviendas principales - Tipo de infraestructura o servicio',
               'Viviendas principales - Estado de conservación',
               'Viviendas principales - Accesibilidad',
               'Viviendas principales - Tipo de instalación',
               'Viviendas principales - Número de plazas de garaje',
               'Viviendas principales - Tipo de dispositivo de energía renovable']

    headers2 = ['Viviendas principales - Estado de conservación ~ Bueno',
                'Viviendas principales - Estado de conservación ~ Excelente',
                'Viviendas principales - Estado de conservación ~ Malo',
                'Viviendas principales - Estado de conservación ~ Muy bueno',
                'Viviendas principales - Estado de conservación ~ Normal',
                'Viviendas principales - Estado de conservación ~ Regular'
               ]

    headers3 = ['Viviendas principales - Tipo de problemática en la zona ~ Contaminación o malos olores (%)',
                'Viviendas principales - Tipo de problemática en la zona ~ Delincuencia (%)',
                'Viviendas principales - Tipo de problemática en la zona ~ Malas comunicaciones (%)',
                'Viviendas principales - Tipo de problemática en la zona ~ Molestias relacionadas con actividades turísticas o locales de hostelería (%)',
                'Viviendas principales - Tipo de problemática en la zona ~ Poca limpieza en las calles (%)',
                'Viviendas principales - Tipo de problemática en la zona ~ Pocas zonas verdes (%)',
                'Viviendas principales - Tipo de problemática en la zona ~ Ruidos exteriores (%)'
               ]

    result = tx.translate_taxonomy_reasoning("es", "en", headers)
    print(result)


def main():
    translate()


if __name__ == "__main__":
    main()
