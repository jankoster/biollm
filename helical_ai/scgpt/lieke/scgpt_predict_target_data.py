from scgpt_train_and_finetune import *

# %%
final_task = "target" # "evaluation" or "target"
if final_task == "target":
    print(filename_target)
    adata_target = sc.read_h5ad(filename_target)
    adata_target.X = adata_target.raw.X.copy()
    adata_target.var["gene_name"] = adata_target.var_names # "Data must have the provided key 'gene_name' in its 'var' section to be processed by the Helical RNA model."
   

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
if final_task == "target":
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
plt.savefig(filename_out_base + "reference_celltype_composition_per_sample.png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close('all')

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
plt.savefig(filename_out_base + "target_celltype_composition_per_sample.png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close('all')

print("Done with dataset:", filename_target)