import torch
import torch.nn as nn

# --------------------------------------------- copy layers ---------------------------------------------

# methode die ich in meinem training verwende !!!!!!

def copy_backbone_and_extra_layers(source_model, target_model, conv_layer_name_source, conv_layer_name_target, pool_layer_name_source, pool_layer_name_target, fc_layer_name_source, fc_layer_name_target, copy_fc):
    """
    Kopiert die Parameter (Weights und Biases) aller Layer im Backbone, sowie eines Conv-Layers und eines Pool-Layers
    von einem Quellmodell zu einem Zielmodell.

    :param source_model: Das Modell, aus dem die Parameter kopiert werden sollen.
    :param target_model: Das Modell, in das die Parameter kopiert werden sollen.
    :param conv_layer_name_source: Der Name des Conv-Layers im Quellmodell.
    :param conv_layer_name_target: Der Name des Conv-Layers im Zielmodell.
    :param pool_layer_name_source: Der Name des Pool-Layers im Quellmodell.
    :param pool_layer_name_target: Der Name des Pool-Layers im Zielmodell.
    """
    
    # 1. Kopiere alle Layer im Backbone
    for layer_name, source_layer in dict(source_model.named_modules()).items():
        if 'backbone' in layer_name:  # Anpassen je nach Struktur des Modells
            target_layer_name = layer_name.replace('backbone', 'backbone')
            if target_layer_name in dict(target_model.named_modules()):
                target_layer = dict(target_model.named_modules())[target_layer_name]
                if isinstance(source_layer, nn.Module) and isinstance(target_layer, nn.Module):
                    target_layer.load_state_dict(source_layer.state_dict())
                else:
                    raise ValueError(f"Die Layer {layer_name} und {target_layer_name} sind nicht kompatibel oder existieren nicht.")
    
    # 2. Kopiere den spezifizierten Conv-Layer
    source_conv_layer = dict(source_model.named_modules())[conv_layer_name_source]
    target_conv_layer = dict(target_model.named_modules())[conv_layer_name_target]
    
    if isinstance(source_conv_layer, nn.Conv2d) and isinstance(target_conv_layer, nn.Conv2d):
        target_conv_layer.load_state_dict(source_conv_layer.state_dict())
    else:
        raise ValueError(f"Die Conv-Layer {conv_layer_name_source} und {conv_layer_name_target} sind nicht kompatibel oder existieren nicht.")
    
    # 3. Kopiere den spezifizierten Pool-Layer
    source_pool_layer = dict(source_model.named_modules())[pool_layer_name_source]
    target_pool_layer = dict(target_model.named_modules())[pool_layer_name_target]
    
    if isinstance(source_pool_layer, nn.Module) and isinstance(target_pool_layer, nn.Module):
        target_pool_layer.load_state_dict(source_pool_layer.state_dict())
    else:
        raise ValueError(f"Die Pool-Layer {pool_layer_name_source} und {pool_layer_name_target} sind nicht kompatibel oder existieren nicht.")
    
    # 4. Kopiere die spezifizierten FC-Layer
    if copy_fc:
        source_fc_layer = dict(source_model.named_modules())[fc_layer_name_source]
        target_fc_layer = dict(target_model.named_modules())[fc_layer_name_target]

        if isinstance(source_fc_layer, nn.Module) and isinstance(target_fc_layer, nn.Module):
            target_fc_layer.load_state_dict(source_fc_layer.state_dict())
        else:
            raise ValueError(f"Die FC-Layer {fc_layer_name_source} und {fc_layer_name_target} sind nicht kompatibel oder existieren nicht.")

# ------------------------------------------------------------------------------------------
# ====================================== test skript: ======================================
# ------------------------------------------------------------------------------------------

def print_parameters(layer, name):
    """ Hilfsfunktion zum Drucken der Parameter eines Layers """
    print(f"--- {name} ---")
    print("Weights:", layer.weight.data)
    if layer.bias is not None:
        print("Biases:", layer.bias.data)
    else:
        print("Biases: None")
    print()

def copy_layer_parameters(source_model, target_model, layer_name_source, layer_name_target):
    """
    Kopiert die Parameter (Weights und Biases) von einem spezifizierten Layer eines Quellmodells 
    zu einem spezifizierten Layer eines Zielmodells.

    :param source_model: Das Modell, aus dem die Parameter kopiert werden sollen.
    :param target_model: Das Modell, in das die Parameter kopiert werden sollen.
    :param layer_name_source: Der Name des Layers im Quellmodell.
    :param layer_name_target: Der Name des Layers im Zielmodell.
    """
    
    # Hole die spezifizierten Layer aus beiden Modellen
    source_layer = dict(source_model.named_modules())[layer_name_source]
    target_layer = dict(target_model.named_modules())[layer_name_target]
    
    # Drucke die Parameter vor dem Kopieren
    print_parameters(source_layer, f"Source Layer ({layer_name_source})")
    print_parameters(target_layer, f"Target Layer Before Copying ({layer_name_target})")
    
    # Überprüfe, ob die Layer kompatibel sind
    if isinstance(source_layer, nn.Module) and isinstance(target_layer, nn.Module):
        # Kopiere die Weights und Biases (falls vorhanden)
        target_layer.weight.data = source_layer.weight.data.clone()
        if source_layer.bias is not None:
            target_layer.bias.data = source_layer.bias.data.clone()
        else:
            target_layer.bias = None
    else:
        raise ValueError(f"Die Layer {layer_name_source} und {layer_name_target} sind nicht kompatibel oder existieren nicht.")
    
    # Drucke die Parameter nach dem Kopieren
    print_parameters(target_layer, f"Target Layer After Copying ({layer_name_target})")

if __name__ == "__main__":
    # Beispielhafte Definition zweier Modelle (architektur sollte bekannt sein)
    source_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    target_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    print("--- source model: ---")
    print(source_model)
    print("--- target model: ---")
    print(target_model)
    
    # Beispiel: Kopieren der Parameter vom ersten Linear-Layer
    copy_layer_parameters(source_model, target_model, '0', '0')  # Layer '0' ist der erste nn.Linear in beiden Modellen
