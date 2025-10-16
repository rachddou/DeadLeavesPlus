import yaml
import os
from itertools import product
import copy

def generate_config_files():
    """
    Génère des fichiers de configuration YAML en combinant différents paramètres.
    Crée un fichier pour chaque combinaison de configurations spécifiées.
    """
    
    # Configuration de base
    base_config = {
        'defaults': [{'override': 'hydra/launcher: joblib'}],
        'shape': {
            'rmin': 10,
            'rmax': 500,
            'alpha': 3.0,
            'shape_type': "poly",
            'multiple_shapes': False
        },
        'task': 1,
        'texture': {
            'texture': True,
            'texture_types': ["sin", "freq_noise", "texture_mixes"],
            'texture_type_frequency': [0.2, 0.6, 0.2],
            'slope_range': [0.5, 2.5],
            'texture_gen': True,
            'warp': True,
            'rdm_phase': False,
            'texture_path': "",
            'perspective': False
        },
        'color': {
            'natural': True,
            'color_path': "/Users/raphael/Workspace/telecom/code/exploration_database_and_code/pristine_images/",
            'grey': False,
            'partial_images': False
        },
        'io': {
            'path_origin': "datasets/",
            'path': "vibrant_leaves/"
        },
        'post_process': {
            'downscaling': True,
            'dof': True,
            'blur_type': "lens",
            'blur': False
        },
        'number': 10,
        'size': 500,
        'image_type': "dead_leaves",
        'test': False
    }
    
    # Paramètres variables pour générer différentes configurations
    variations = {
        'shape_type': ['poly', 'disks', 'rectangles'],
        'texture_enabled': [True, False],
        'rmin': [5,10,20,50,100,200],
        'texture_type_frequency':[[0.,0.9,0.1],
                                  [0.,0.75,0.25],
                                  [0.,1.,0.],
                                  [0.25,0.75,0.],
                                  [0.1,0.9,0.],
                                  [0.15,0.7,0.15]],
        'slope_range': [[[0.5,2.375]],
                        [[0.5,1.125],[1.75,2.375]],
                        [1.125,2.375],
                        [0.5,1.125],
                        [1.125,1.75],
                        [1.75,2.375]]
        
    }
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = "generated_configs"
    os.makedirs(output_dir, exist_ok=True)
    
    config_count = 0
    
    # Générer toutes les combinaisons possibles
    keys = list(variations.keys())
    values = [variations[key] for key in keys]
    
    for combination in product(*values):
        config = copy.deepcopy(base_config)
        config_name_parts = []
        
        # Appliquer les variations
        for i, key in enumerate(keys):
            value = combination[i]
            config_name_parts.append(f"{key}_{value}")
            
            if key == 'shape_type':
                config['shape']['shape_type'] = value
            elif key == 'texture_enabled':
                config['texture']['texture'] = value
            elif key == 'rmin':
                config['shape']['rmin'] = value
            elif key == 'texture_type_frequency':
                config['texture']['texture_type_frequency'] = value
            elif key == 'slope_range':
                config['texture']['slope_range'] = value
        
        # Nom du fichier basé sur la configuration
        config_name = "_".join(config_name_parts)
        
        config['io']['path'] = f"vibrant_leaves/{config_name}/"
        
        filename = f"config_{config_count:03d}_{config_name}.yaml"
        filepath = os.path.join(output_dir, filename)
        
        # Sauvegarder le fichier YAML
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        config_count += 1
        print(f"Généré: {filename}")
    
    print(f"\nTotal: {config_count} fichiers de configuration générés dans '{output_dir}/'")

def generate_specific_configs():
    """
    Génère des configurations spécifiques pour des cas d'usage particuliers.
    """
    base_config = {
        'defaults': [{'override': 'hydra/launcher: joblib'}],
        'shape': {
            'rmin': 10,
            'rmax': 1000,
            'alpha': 3.0,
            'shape_type': "poly",
            'multiple_shapes': False
        },
        'task': 1,
        'texture': {
            'texture': True,
            'texture_types': ["sin", "freq_noise", "texture_mixes"],
            'texture_type_frequency': [0.2, 0.6, 0.2],
            'slope_range': [0.5, 2.5],
            'texture_gen': True,
            'warp': True,
            'rdm_phase': False,
            'texture_path': "",
            'perspective': False
        },
        'color': {
            'natural': True,
            'color_path': "/Users/raphael/Workspace/telecom/code/exploration_database_and_code/pristine_images/",
            'grey': False,
            'partial_images': False
        },
        'io': {
            'path_origin': "datasets/",
            'path': "vibrant_leaves/"
        },
        'post_process': {
            'downscaling': True,
            'dof': False,
            'blur_type': "None",
            'blur': False
        },
        'number': 10,
        'size': 1000,
        'image_type': "dead_leaves",
        'test': False
    }
    
    # Configurations spécifiques
    specific_configs = [
        {
            'name': 'high_res_natural',
            'changes': {
                'size': 2048,
                'number': 100,
                'color.natural': True,
                'texture.texture': True
            }
        },
        {
            'name': 'small_grey_test',
            'changes': {
                'size': 256,
                'number': 5,
                'color.natural': False,
                'color.grey': True,
                'test': True
            }
        },
        {
            'name': 'circle_blur',
            'changes': {
                'shape.shape_type': 'circle',
                'post_process.blur': True,
                'post_process.blur_type': 'gaussian'
            }
        }
    ]
    
    output_dir = "specific_configs"
    os.makedirs(output_dir, exist_ok=True)
    
    for spec_config in specific_configs:
        config = copy.deepcopy(base_config)
        
        # Appliquer les changements
        for key, value in spec_config['changes'].items():
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value
        
        filename = f"{spec_config['name']}.yaml"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Généré: {filename}")

if __name__ == "__main__":
    print("Génération des fichiers de configuration...")
    generate_config_files()
    # print("\nGénération des configurations spécifiques...")
    # # generate_specific_configs()
    # # print("Terminé!")