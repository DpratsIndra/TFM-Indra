import json
import os

def generate_mapping(output_path: str = "data/eval_datasets/APT_REPORT/mitre_group_mapping.json"):
    """
    Descarga la base de datos oficial STIX de MITRE ATT&CK (Enterprise) 
    y genera un diccionario mapeando todos los aliases de grupos APT 
    a sus técnicas históricamente utilizadas.
    """
    print("[*] Leyendo base de datos STIX local de MITRE ATT&CK Enterprise...")
    local_stix_path = "data/mitre_data/enterprise-attack.json"
    
    if not os.path.exists(local_stix_path):
        print(f"[!] Error: El archivo {local_stix_path} no existe. Por favor, corre primero la fase de indexación o descárgalo.")
        return
        
    try:
        with open(local_stix_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[!] Error leyendo los datos locales de MITRE: {e}")
        return

    groups = {}
    relationships = []

    print("[*] Parseando objetos STIX (Grupos y Relaciones)...")
    for obj in data.get("objects", []):
        if obj["type"] == "intrusion-set":
            # Guardar el nombre del grupo y sus alias
            aliases = [obj.get("name", "").lower()] + [a.lower() for a in obj.get("aliases", [])]
            groups[obj["id"]] = {"name": obj["name"], "aliases": aliases, "techniques": set()}
        elif obj["type"] == "relationship" and obj["relationship_type"] == "uses":
            relationships.append(obj)

    # Mapa para traducir un ID interno de STIX a un ID de MITRE (ej. T1059)
    tech_map = {
        obj["id"]: obj.get("external_references", [{}])[0].get("external_id") 
        for obj in data.get("objects", []) 
        if obj["type"] == "attack-pattern"
    }

    # Asociar técnicas a los grupos usando las relaciones 'uses'
    for rel in relationships:
        if rel["source_ref"] in groups and rel["target_ref"] in tech_map:
            tech_id = tech_map[rel["target_ref"]]
            if tech_id:
                groups[rel["source_ref"]]["techniques"].add(tech_id)

    # Aplanar el diccionario: Key=Alias, Value=[Técnicas]
    mapping = {}
    for g in groups.values():
        for alias in g["aliases"]:
            if alias: # Evitar strings vacíos
                mapping[alias] = list(g["techniques"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)
        
    print(f"[*] Mapeo creado en {output_path} con {len(mapping)} alias de grupos APT resolubles.")

if __name__ == "__main__":
    generate_mapping()
