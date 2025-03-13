import time
import torch
import torch.optim as optim

from rnaglib.tasks.RNA_VS.task import VirtualScreening
from rnaglib.transforms import GraphRepresentation
from rnaglib.transforms import FeaturesComputer

# Create a task
framework = 'pyg'
ef_task = VirtualScreening('RNA_VS', ligand_framework=framework)

# Build corresponding datasets and dataloader
features_computer = FeaturesComputer(nt_features=['nt_code'])
representations = [GraphRepresentation(framework=framework)]
rna_dataset_args = {'representations': representations, 'features_computer': features_computer}
rna_loader_args = {'batch_size': 16, 'shuffle': True, 'num_workers': 0}
train_dataloader, val_dataloader, test_dataloader = ef_task.get_split_loaders(dataset_kwargs=rna_dataset_args,
                                                                              dataloader_kwargs=rna_loader_args)

# Create an encoding model.
# This model must implement a predict_ligands(pocket, ligands) method
if framework == 'pyg':
    from rnaglib.tasks.RNA_VS.model_pyg import RNAEncoder, LigandGraphEncoder, Decoder, VSModel
else:
    from rnaglib.tasks.RNA_VS.model_dgl import RNAEncoder, LigandGraphEncoder, Decoder, VSModel

model = VSModel(encoder=RNAEncoder(), lig_encoder=LigandGraphEncoder(), decoder=Decoder())
assert hasattr(model, 'predict_ligands') and callable(getattr(model, 'predict_ligands'))

# Train
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()
epochs = 10
t0 = time.time()
for k in range(epochs):
    for i, batch in enumerate(train_dataloader):
        pockets = batch['pocket']
        ligands = batch['ligand']
        actives = torch.tensor(batch['active'], dtype=torch.float32)

        optimizer.zero_grad()
        out = model(pockets, ligands)
        loss = criterion(input=torch.flatten(out), target=actives)
        loss.backward()
        optimizer.step()
        # if i > 3:
        #     break
        if not i % 5:
            print(f'Epoch {k}, batch {i}/{len(train_dataloader)}, '
                  f'loss: {loss.item():.4f}, time: {time.time() - t0:.1f}s')

model = model.eval()
final_vs = ef_task.evaluate(model)
