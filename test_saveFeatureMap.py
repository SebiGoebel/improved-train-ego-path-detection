import torch

# Beispielhafte Daten: 5 zweidimensionale PyTorch-Tensoren erstellen
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
tensor3 = torch.tensor([[13, 14, 15], [16, 17, 18]])
tensor4 = torch.tensor([[19, 20, 21], [22, 23, 24]])
tensor5 = torch.tensor([[25, 26, 27], [28, 29, 30]])

# Speichere alle 5 2D-Tensoren als .pt Datei mit torch.save
torch.save({'tensor1': tensor1, 'tensor2': tensor2, 'tensor3': tensor3, 'tensor4': tensor4, 'tensor5': tensor5}, 'five_2d_tensors.pt')

# Laden der Tensoren
loaded_tensors = torch.load('five_2d_tensors.pt')
tensor1_loaded = loaded_tensors['tensor1']
tensor2_loaded = loaded_tensors['tensor2']
tensor3_loaded = loaded_tensors['tensor3']
tensor4_loaded = loaded_tensors['tensor4']
tensor5_loaded = loaded_tensors['tensor5']

# Ausgabe der geladenen Tensoren
print("Tensor 1:", tensor1_loaded)
print("Tensor 2:", tensor2_loaded)
print("Tensor 3:", tensor3_loaded)
print("Tensor 4:", tensor4_loaded)
print("Tensor 5:", tensor5_loaded)
