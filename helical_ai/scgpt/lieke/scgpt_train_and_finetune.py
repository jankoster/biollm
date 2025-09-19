
# %% [markdown]
# To run this script from the command line in conda singlecellgpt_env:
# python cell_type_annotation_gpt_raymond.py 2503_kr113a_10k --tee
#
# Each run creates several output files and a unique log like output_2503_kr113a_10k_20250917_143210.log
# with --tee it also prints to terminal
#
# # Cell type annotation prediction - Jansky & Westerhout
# 
# 
# In this notebook, an [scGPT](https://www.nature.com/articles/s41592-024-02201-0) model is used to predict a cell type annotation with a given gene expression profile.
# 
# This follows the tutorial from scGPT [here](https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Annotation.ipynb), but instead of fine-tuning the entire model, a smaller neural network is trained, using the embeddings of the gene expressions as inputs, to make a prediction.
# 
# The same approach is made with the [Geneformer](https://www.nature.com/articles/s41586-023-06139-9.epdf?sharing_token=u_5LUGVkd3A8zR-f73lU59RgN0jAjWel9jnR3ZoTv0N2UB4yyXENUK50s6uqjXH69sDxh4Z3J4plYCKlVME-W2WSuRiS96vx6t5ex2-krVDS46JkoVvAvJyWtYXIyj74pDWn_DutZq1oAlDaxfvBpUfSKDdBPJ8SKlTId8uT47M%3D) model and the results are compared against each other.
# 
# This approach greatly reduces time and complexity.

# %%
#!pip3 install helical
#!conda install -c conda-forge louvain
#!pip3 install datasets --upgrade

# %%
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim
from helical.models.scgpt import scGPT, scGPTConfig
from copy import deepcopy
from torch.nn.functional import one_hot
import scanpy as sc

# %% [markdown]
# Get settings
from scgpt_settings import *

# %%
# Load fine-tuning data
adata = sc.read_h5ad(filename_ref)


# %%
# Seurat to AnnData conversion can be tricky. Ensure that the raw counts are properly assigned. 
print(adata.X)
print(adata.raw.X)

# %%
mask = adata.obs[ref_meta_colname].isin(categories_of_interest)
print(f"Number of categories of interest (-related) cells: {mask.sum()}")



# %%
# Only copy subset if needed
adata = adata[mask].copy()

# %%
# Copy raw counts to adata.X if necessary
adata.X = adata.raw.X.copy()
adata.var["gene_name"] = adata.var_names # "Data must have the provided key 'gene_name' in its 'var' section to be processed by the Helical RNA model."

# %%
adata.obs

# %% [markdown]
# Randomly choose approx. 20% of patients to leave out for test set (final evaluation).

# %%
import random

  # unique patient ids
unseen_patients = random.sample(list(set(adata.obs[sample_id])), 3)
print(unseen_patients)

adata_evaluation = adata[adata.obs[sample_id].isin(unseen_patients)]
adata_finetuning  = adata[~adata.obs[sample_id].isin(unseen_patients)]


# %% [markdown]
# We are interested in the names of the cells we want to predict. They are saved in `adata.obs[ref_meta_colname]`.
# 
# Additionally, we need to know how many distinct cell types/classes we have.

# %%
adata_finetuning.obs


# %%
adata_finetuning.var

# %%
# get labels: the celltype
num_types = adata_finetuning.obs[ref_meta_colname].unique().shape[0]
id2type = dict(enumerate(adata_finetuning.obs[ref_meta_colname].astype("category").cat.categories))

celltypes_labels = np.array(adata_finetuning.obs[ref_meta_colname].tolist())

# %% [markdown]
# This is all summarized in this dictionary:

# %%
id2type

# %% [markdown]
# Use the Helical package to get the embeddings of the gene expression profile.
# 
# The only thing we need to specify is the column containing the names of the genes. (`gene_name` in this case)
# 
# The resulting embeddings are the input features `x` for our smaller NN model.

# %% [markdown]
# # scGPT

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure and initialize the scGPT model
scgpt_config = scGPTConfig(batch_size=50, device=device)
scgpt = scGPT(configurer = scgpt_config)

# Process the data for the scGPT model
Normalize_SubsetHighlyVariable = False # This logic is also connected to processing the left out data (adata_unseen).

if Normalize_SubsetHighlyVariable:
    data_processed = scgpt.process_data(adata_finetuning, gene_names = "gene_name", fine_tuning=True)
else:
    data_processed = scgpt.process_data(adata_finetuning, gene_names = "gene_name")

# Get embeddings
x_scgpt = scgpt.get_embeddings(data_processed)
x_scgpt.shape

# %% [markdown]
# With the input features, we also need the corresponding labels `y`.
# 
# They correspond to the cell type labels.
# 
# As we have a categorical prediction, we transform the cell type labels to integer labels to work with CrossEntropyLoss later.

# %%
y = celltypes_labels

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_encoded = one_hot(torch.tensor(y_encoded),num_types).float()

# %%
freq = pd.Series(y).value_counts()
freq

# %%
#np.save( filename_root_ref + "x_scgpt",x_scgpt)
x_scgpt_reloaded = np.load(filename_root_ref + "x_scgpt.npy")
#np.array_equal(x_scgpt_reloaded, x_scgpt, equal_nan=True)

# %% [markdown]
# ## Define and train the model

# %%
input_shape = 512

# Define the model architecture
head_model = nn.Sequential(
    nn.Linear(input_shape, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(32, num_types)
    )

print(head_model)

# %%
def train_model(model: nn.Sequential,
                X_train: torch.Tensor,
                y_train: torch.Tensor,
                X_val: torch.Tensor,
                y_val: torch.Tensor,
                optimizer = optim.Adam,
                loss_fn = nn.CrossEntropyLoss(),
                num_epochs = 100,
                batch = 64):

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    # Validation dataset
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # Ensure model is in training mode
    model.train()

    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)

            # Compute loss
            loss = loss_fn(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase (optional)
        model.eval()
        with torch.no_grad():
            val_losses = []
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_loss = loss_fn(val_outputs, val_y)
                val_losses.append(val_loss.item())

            print(f"Epoch {epoch+1}, Validation Loss: {sum(val_losses)/len(val_losses)}")

        # Set back to training mode for next epoch
        model.train()

    model.eval()
    return model

# %%
X_train, X_test, y_train, y_test = train_test_split(x_scgpt, y_encoded, test_size=0.1, random_state=42)

head_model_scgpt = deepcopy(head_model)
head_model_scgpt = train_model(head_model_scgpt,
                               torch.from_numpy(X_train),
                               y_train,
                               torch.from_numpy(X_test),
                               y_test,
                               optim.Adam(head_model_scgpt.parameters(), lr=0.001),
                               nn.CrossEntropyLoss())

# path to save the model
model_path = os.path.join(foldername_output, timestamp + "_head_model_scgpt.pt")

# save the state dict
torch.save(head_model_scgpt.state_dict(), model_path)

print(f"Trained model saved to: {model_path}")

# %%
# Predictions on the test set and ground truth
predictions_nn = head_model_scgpt(torch.Tensor(X_test))
y_pred = np.array(torch.argmax(predictions_nn, dim=1))
y_true = np.array(y_test.argmax(axis=1))

# %% [markdown]
# ## Present the results
# - on the test set and,
# - a separate, unseen evaluation set

# %%
def get_evaluations(name_data_set, y_true, y_pred) -> dict:
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='macro')
  f1 = f1_score(y_true, y_pred, average='macro')
  recall = recall_score(y_true, y_pred, average='macro')
  print(f"{name_data_set} accuracy: {(accuracy*100):.1f}%")
  print(f"{name_data_set} precision: {(precision*100):.1f}%")
  print(f"{name_data_set} f1: {(f1*100):.1f}%")
  print(f"{name_data_set} recall: {(recall*100):.1f}%")
  return {
      "accuracy": accuracy,
      "precision": precision,
      "f1": f1,
      "recall": recall,
  }

# %%
get_evaluations("Test set", y_true, y_pred)

# %%
#get_evaluations_2("Test set", y_true, y_pred)

# %%
# Visualize class distribution
from sklearn_evaluation import plot
ax = plot.target_analysis(y_true)  # returns Axes
ax.legend(id2type.values(), title="Cell Types", loc="upper right")

fig = ax.get_figure()              # get the parent Figure
fig.savefig(filename_out_base + "reftest_true_celltype_frequency.png", dpi=300, bbox_inches="tight")
print(id2type)

# %%
## Only if you want to forget about zero_devision
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def get_evaluations_2(name, y_true, y_pred):
#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
#     rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
#     f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
#     print(f"{name} accuracy: {acc*100:.2f}%")
#     print(f"{name} precision: {prec*100:.2f}%")
#     print(f"{name} recall: {rec*100:.2f}%")
#     print(f"{name} F1: {f1*100:.2f}%")

# %% [markdown]
# Load the unseen evaluation set. Two options:
# 
# 1. Evaluation: Load the left out evaluation set from the finetuning dataset. (with true labels - Jansky)
# 2. Target: Load the dataset that the predictions will be made on. (no true labels - Westerhout)

# %%
final_task = "evaluation" # "evaluation" or "target"
if final_task == "evaluation": 
    adata_unseen = adata_evaluation # AnnData preprocessing was done before the finetuning - evaluation split.
    

# %%
# Process the unseen data
if Normalize_SubsetHighlyVariable:
    data_processed = scgpt.process_data(adata_unseen, gene_names = "gene_name", fine_tuning=True)
else:
    data_processed = scgpt.process_data(adata_unseen, gene_names = "gene_name")

# Get embeddings and predictions
x_unseen = scgpt.get_embeddings(data_processed)
predictions_nn_unseen = head_model_scgpt(torch.Tensor(x_unseen))

# %%
if final_task == "evaluation":
    y_pred_unseen = np.array(torch.argmax(predictions_nn_unseen, dim=1))
    y_true_unseen = np.array(adata_unseen.obs[ref_meta_colname].astype("category").cat.codes)
    get_evaluations("Unseen evaluation set", y_true_unseen, y_pred_unseen)
    fig = plot.target_analysis(y_true_unseen)
    fig.suptitle("Unseen evaluation set")  # optional title
    fig.savefig(filename_out_base + "target_analysis_unseen.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(id2type)

# %%
if final_task == "evaluation":
    num_types = adata_unseen.obs[ref_meta_colname].unique().shape[0]
    id2type_unseen = dict(enumerate(adata_unseen.obs[ref_meta_colname].astype("category").cat.categories))
    print(sorted(id2type_unseen) == sorted(id2type))
    
print(id2type)
print(id2type_unseen)

# %%  
if final_task == "evaluation":
    y_true_unseen = np.array(adata_unseen.obs[ref_meta_colname].tolist())
    y_pred_unseen = [id2type[prediction] for prediction in np.array(torch.argmax(predictions_nn_unseen, dim=1))]

    scgpt_results = get_evaluations("Evaluation set", y_true_unseen, y_pred_unseen)

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define all possible labels in the correct order
all_labels = list(id2type.values())  # includes CCL19+ fibroblast even if missing

df_eval = pd.DataFrame({
    "true": y_true_unseen,
    "pred": y_pred_unseen
})

print(df_eval.groupby(["true", "pred"]).size())

# Compute confusion matrix, forcing all labels
cm = confusion_matrix(y_true_unseen, y_pred_unseen, labels=all_labels)

# Plot with seaborn, keeping label order
sns.heatmap(cm, annot=True, fmt="d", xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix (unseen set)")
plt.savefig(filename_out_base + "confusionmatrix_unseenset_celltype_composition_per_sample.png", dpi=300, bbox_inches="tight")
# plt.show()
plt.close('all')

# %%
# Unique labels in the training set
train_labels = adata.obs[ref_meta_colname]
print("Training labels:", sorted(train_labels.unique()))

# Unique labels in the unseen set
unseen_labels = adata_unseen.obs[ref_meta_colname]
print("Unseen labels:", sorted(unseen_labels.unique()))

# %%
## if you want to work with common_labels only
# common_labels = list(set(train_labels.unique()).intersection(unseen_labels.unique()))
# print("Labels used for evaluation:", sorted(common_labels))

# # Subset training and unseen labels to common labels
# y_true_unseen = adata_unseen.obs[ref_meta_colname].astype("category")
# y_true_unseen = y_true_unseen.cat.set_categories(common_labels).cat.codes.to_numpy()

# y_pred_unseen = np.array(torch.argmax(predictions_nn_unseen, dim=1))

# %%
id2type

train_counts = adata[adata.obs[ref_meta_colname].isin(categories_of_interest)].obs[ref_meta_colname].value_counts()
print(train_counts)

# %% [markdown]
# We should double check that the cell types are mapped to the correct id numbers for both the training data and this new data set.

# %%
if final_task == "evaluation":
    num_types = adata_unseen.obs[ref_meta_colname].unique().shape[0]
    id2type_unseen = dict(enumerate(adata_unseen.obs[ref_meta_colname].astype("category").cat.categories))
    print(id2type_unseen == id2type)

   # print("Common labels:", sorted(common_labels))
    print("Unseen categories:", list(adata_unseen.obs[ref_meta_colname].astype("category").cat.categories))

# %%
if final_task == "evaluation":
    num_types = adata_unseen.obs[ref_meta_colname].unique().shape[0]
    id2type_unseen = dict(enumerate(adata_unseen.obs[ref_meta_colname].astype("category").cat.categories))
    print(sorted(id2type_unseen) == sorted(id2type))
    print(id2type_unseen)
    print(id2type)

# %%
# Convert predicted indices to labels safely
y_pred_indices = np.array(torch.argmax(predictions_nn_unseen, dim=1))
y_pred_unseen = []

for idx in y_pred_indices:
    if idx in id2type:
        y_pred_unseen.append(id2type[idx])
    else:
        y_pred_unseen.append("unknown")  # or np.nan


unique_preds = set(y_pred_unseen)
print(unique_preds)


#print(unique_trues)

y_pred_unseen


# %%
unique_trues = set(y_true_unseen)
print(unique_trues)

y_true_unseen

# %%
if final_task == "evaluation":
    y_true_unseen = np.array(adata_unseen.obs[ref_meta_colname].tolist())
    y_pred_unseen = [id2type[prediction] for prediction in np.array(torch.argmax(predictions_nn_unseen, dim=1))]

    scgpt_results = get_evaluations("Evaluation set", y_true_unseen, y_pred_unseen)

# %% [markdown]
# Plot a confusion matrix to visualise the classification performance for each the cell type. This is done for the evalation set.

# %%
if final_task == "evaluation":
    set_predicted_cell_types = list(adata_unseen.obs[ref_meta_colname].unique())
    for i in set(y_pred_unseen):
        if i not in set_predicted_cell_types:
            set_predicted_cell_types.remove(i)

    cm = confusion_matrix(y_true_unseen, y_pred_unseen)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(cm, index=set_predicted_cell_types[:cm.shape[0]], columns=set_predicted_cell_types[:cm.shape[1]])
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Normalized Confusion Matrix (Unseen set)")
    plt.tight_layout()
    plt.savefig(filename_out_base + "confusionmatrix_unseenset_normalized.png", dpi=300, bbox_inches="tight")
    plt.close()



