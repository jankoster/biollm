# %% [markdown]
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
import anndata as ad
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from scipy.sparse import lil_matrix
import torch.optim as optim
from helical.models.scgpt import scGPT, scGPTConfig
from helical.models.geneformer import Geneformer, GeneformerConfig
from copy import deepcopy
from torch.nn.functional import one_hot
import scanpy as sc
import os

# %%
filename_root_target = "250215_kr250212a_10k_integrated-cca"
filename_root_ref = "Fskin_obj_2_4_1_webatlas"

foldername_input = "/home/lieke_unix/input/raymond/"
foldername_output = "/home/lieke_unix/output/raymond/"
filename_ref= os.path.join(foldername_input, filename_root_ref + ".h5ad")

filename_out_base = os.path.join(foldername_output, filename_root_target)
filename_target = os.path.join(foldername_input, filename_root_target + ".h5ad")
ref_meta_colname = "annotation_fine"
filename_out_predictions = filename_out_base + "_celltype_preds.csv"
sample_id = "sanger_id"
sample_id_target = "HTO_maxID"
predictions_meta_name = "cell_type_predictions"

# %%
# Define categories of interest (fibroblast example)
categories_of_interest = [
    "CCL19+ fibroblast",
    "FRZB+ early fibroblast",
    "HOXC5+ early fibroblast",
    "Myofibroblasts",
    "PEAR1+ fibroblast",
    "WNT2+ fibroblast",
]

# %% [markdown]
# Fine-tuning data: Jansky

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
plot.target_analysis(y_true)
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
if final_task == "target":
    adata_unseen = sc.read_h5ad(filename_target)
    adata_unseen.X = adata_unseen.raw.X.copy()
    adata_unseen.var["gene_name"] = adata_unseen.var_names # "Data must have the provided key 'gene_name' in its 'var' section to be processed by the Helical RNA model."
elif final_task == "evaluation": 
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
    plot.target_analysis(y_true_unseen)
    print(id2type)
elif final_task == "target":
    save_annData_with_predictions = True

    y_pred_unseen = [id2type[prediction] for prediction in np.array(torch.argmax(predictions_nn_unseen, dim=1))]
    if save_annData_with_predictions:
        adata_unseen.obs[predictions_meta_name] = y_pred_unseen
        adata_unseen.obs.to_csv(filename_out_predictions)

    print(adata_unseen.obs)


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
import pandas as pd

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
plt.show()

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
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    set_predicted_cell_types = list(adata_unseen.obs[ref_meta_colname].unique())
    for i in set(y_pred_unseen):
        if i not in set_predicted_cell_types:
            set_predicted_cell_types.remove(i)

    cm = confusion_matrix(y_true_unseen, y_pred_unseen)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(cm, index=set_predicted_cell_types[:cm.shape[0]], columns=set_predicted_cell_types[:cm.shape[1]])
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")




# %%
final_task = "target" # "evaluation" or "target"
if final_task == "target":
    print(filename_target)
    adata_target = sc.read_h5ad(filename_target)
    adata_target.X = adata_target.raw.X.copy()
    adata_target.var["gene_name"] = adata_target.var_names # "Data must have the provided key 'gene_name' in its 'var' section to be processed by the Helical RNA model."
elif final_task == "evaluation": 
    adata_unseen = adata_evaluation # AnnData preprocessing was done before the finetuning - evaluation split.
    

# %%
# Process the unseen data
if Normalize_SubsetHighlyVariable:
    data_processed = scgpt.process_data(adata_target, gene_names = "gene_name", fine_tuning=True)
else:
    data_processed = scgpt.process_data(adata_target, gene_names = "gene_name")

# Get embeddings and predictions
x_unseen = scgpt.get_embeddings(data_processed)
predictions_nn_unseen = head_model_scgpt(torch.Tensor(x_unseen))

# %%
if final_task == "evaluation":
    y_pred_unseen = np.array(torch.argmax(predictions_nn_unseen, dim=1))
    y_true_unseen = np.array(adata_unseen.obs[ref_meta_colname].astype("category").cat.codes)
    get_evaluations("Unseen evaluation set", y_true_unseen, y_pred_unseen)
    plot.target_analysis(y_true_unseen)
    print(id2type)
elif final_task == "target":
    save_annData_with_predictions = True

    y_pred_unseen = [id2type[prediction] for prediction in np.array(torch.argmax(predictions_nn_unseen, dim=1))]
    if save_annData_with_predictions:
        adata_target.obs[predictions_meta_name] = y_pred_unseen
        adata_target.obs.to_csv(filename_out_predictions)

    print(adata_target.obs)


# %%
celltype_counts = (
    adata.obs
    .groupby([sample_id, ref_meta_colname])
    .size()
    .reset_index(name="count")
)
celltype_freq = (
    celltype_counts
    .groupby(sample_id)
    .apply(lambda df: df.assign(freq=df["count"] / df["count"].sum()))
    .reset_index(drop=True)
)
celltype_freq

import matplotlib.pyplot as plt
# pivot to have patients as rows and cell types as columns (proportions)
celltype_pivot = celltype_freq.pivot(index=sample_id,
                                     columns=ref_meta_colname,
                                     values="freq").fillna(0)
# stacked bar plot
celltype_pivot.plot(kind="bar",
                    stacked=True,
                    figsize=(16,6),
                    colormap="tab20")  # nice color palette
plt.ylabel("Proportion of cell types")
plt.xlabel("Sample")
plt.xticks(rotation=45, ha="right")
plt.title("Cell type composition per sample")
plt.legend(title="Cell type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(filename_out_base + "adata_celltype_composition_per_sample.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
adata_target.obs

# %%
adata_target.obs[sample_id_target].unique

# %%
n_unique_sampleids = adata_target.obs[sample_id_target].nunique()
print("Number of unique sample IDs:", n_unique_sampleids)

# %%
# Proportion of cell types per sample

celltype_counts = (
    adata_target.obs
    .groupby([sample_id_target, predictions_meta_name])
    .size()
    .reset_index(name="count")
)
celltype_freq = (
    celltype_counts
    .groupby(sample_id_target)
    .apply(lambda df: df.assign(freq=df["count"] / df["count"].sum()))
    .reset_index(drop=True)
)
celltype_freq

import matplotlib.pyplot as plt
# pivot to have patients as rows and cell types as columns (proportions)
celltype_pivot = celltype_freq.pivot(index=sample_id_target,
                                     columns=predictions_meta_name,
                                     values="freq").fillna(0)
# stacked bar plot
celltype_pivot.plot(kind="bar",
                    stacked=True,
                    figsize=(n_unique_sampleids,6),
                    colormap="tab20")  # nice color palette
plt.ylabel("Proportion of cell types")
plt.xlabel("Sample")
plt.xticks(rotation=45, ha="right")
plt.title("Cell type composition per sample")
plt.legend(title="Cell type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(filename_out_base + "adata_target_celltype_composition_per_sample.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Geneformer

# %% [markdown]
# Let's do the same with the Geneformer.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
if 'rows' in adata.obs:
    adata_finetuning.obs['rows'] = adata_finetuning.obs['rows'].astype(str)
geneformer_config = GeneformerConfig(batch_size=50, device=device)
geneformer = Geneformer(configurer = geneformer_config)

data_geneformer = geneformer.process_data(adata_finetuning, gene_names = "gene_name")
x_geneformer = geneformer.get_embeddings(data_geneformer)
x_geneformer.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(x_geneformer, y_encoded, test_size=0.1, random_state=42)

head_model_geneformer = deepcopy(head_model)
head_model_geneformer = train_model(head_model_geneformer,
                                    torch.tensor(X_train),
                                    y_train,
                                    torch.tensor(X_test),
                                    y_test,
                                    optim.Adam(head_model_geneformer.parameters(), lr=0.001),
                                    nn.CrossEntropyLoss())

# %%
data_unseen_geneformer = geneformer.process_data(adata_unseen, gene_names = "gene_name")
x_unseen_geneformer = geneformer.get_embeddings(data_unseen_geneformer)
predictions_nn_unseen_geneformer = head_model_geneformer(torch.Tensor(x_unseen_geneformer))

# %%
y_true_unseen = np.array(adata_unseen.obs["LVL1"].tolist())
y_pred_unseen = [id2type[prediction] for prediction in np.array(torch.argmax(predictions_nn_unseen_geneformer, dim=1))]

geneformer_results = get_evaluations("Evaluation set", y_true_unseen, y_pred_unseen)

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

values_1 = [scgpt_results["accuracy"], geneformer_results["accuracy"]]
x = ["scGPT", "Geneformer"]
axs[0, 0].bar(x, values_1, width=0.4)
axs[0, 0].set_title("Accuracy")
axs[0, 0].set_ylim([0, 1])

values_2 = [scgpt_results["precision"], geneformer_results["precision"]]
axs[0, 1].bar(x, values_2, width=0.4)
axs[0, 1].set_title("Precision")
axs[0, 1].set_ylim([0, 1])

values_3 = [scgpt_results["f1"], geneformer_results["f1"]]
axs[1, 0].bar(x, values_3, width=0.4)
axs[1, 0].set_title("F1")
axs[1, 0].set_ylim([0, 1])

values_4 = [scgpt_results["recall"], geneformer_results["recall"]]
axs[1, 1].bar(x, values_4, width=0.4)
axs[1, 1].set_title("Recall")
axs[1, 1].set_ylim([0, 1])

fig.suptitle("scGPT vs. Geneformer \n Probing Comparison")
fig.tight_layout()
plt.savefig(filename_out_base + "scGPT_vs_Geneformer.png", dpi=300, bbox_inches="tight")
plt.show()


# %% [markdown]
# 
# ## scGPT
# - Accuracy: 99.2%
# - Precision: 90.8%
# - Recall: 79.1%
# - Macro F1: 80.7%
# 
# ## Geneformer
# - Accuracy: 98.9%
# - Precision: 71.6%
# - Recall: 73.6%
# - Macro F1: 77.3%

# %% [markdown]
#  (c) Helical 2024 - Developed by the Helical Team


