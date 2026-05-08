#!/bin/bash

# pip install tensorboardX
# pip install h5py
# pip install opencv-python
# pip install scikit-image
# pip install numba
# pip install hydra-core
# pip install hydra-joblib-launcher --upgrade


# cd /scratch/Raphael/DeadLeavesPlus/


# pip install -r requirements.txt


# command1="python image_generation.py --multirun task=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40 --config-name $config_file"

# eval "$command1"


# Boucle sur les fichiers de configuration de 0 à 99
# Vérifier que les paramètres de début et de fin sont fournis
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start> <end>"
    exit 1
fi

start=$1
end=$2

for i in $(seq "$start" "$end"); do
    # Formater le numéro avec 3 chiffres (000, 001, 002, etc.)
    config_num=$(printf "%03d" $i)
    
    # Trouver le fichier de configuration correspondant
    config_file=$(find config/generated_configs/ -name "config_${config_num}_*.yaml" | head -1)
    
    # Vérifier si le fichier existe
    if [ -f "$config_file" ]; then
        # Extraire le nom du fichier sans le chemin et l'extension
        config_name=$(basename "$config_file" .yaml)
        
        echo "========================================="
        echo "Traitement de la configuration $i: $config_name"
        echo "========================================="
        
        # command1="python image_generation.py --multirun task=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40 --config-path=config/generated_configs --config-name=$config_name"
        
        # eval "$command1"
        
        # echo "Configuration $i terminée"
        # echo ""
    else
        echo "⚠️  Fichier de configuration config_${config_num}_*.yaml non trouvé, passage au suivant..."
    fi
done


echo "========================================="
echo "Toutes les configurations ont été traitées!"
echo "========================================="