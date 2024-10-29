import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast,GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import scanpy as sc
from typing import Optional
from torch.utils.data import DataLoader,Dataset
class CLIPEncoder(nn.Module):
   def __init__(self,config):
       super().__init__()
       self.layers=nn.ModuleList([nn.Linear(config.hidden_size,config.hidden_size) for _ in range(config.num_hidden_layers)])
       self.layernorm=nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
   def forward(self,x):
       for layer in self.layers:x=F.relu(layer(x))
       return self.layernorm(x)
class ProjectionHead(nn.Module):
   def __init__(self,input_dim,output_dim,hidden_dim=None,dropout=0.1):
       super().__init__()
       if hidden_dim is None:hidden_dim=input_dim
       self.projection=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,output_dim),nn.LayerNorm(output_dim))
   def forward(self,x):return self.projection(x)
class OptimizedProjectionHead(nn.Module):
   def __init__(self,input_dim,output_dim,hidden_dim=None,dropout=0.1):
       super().__init__()
       if hidden_dim is None:hidden_dim=input_dim*2
       self.skip=nn.Linear(input_dim,output_dim)
       self.layer_scale=nn.Parameter(torch.ones(1)*1e-4)
       self.projection=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,output_dim),nn.LayerNorm(output_dim))
   def forward(self,x):return self.skip(x)+self.layer_scale*self.projection(x)
class DiffMapProteinCLIPModule(nn.Module):
   def __init__(self,config):
       super().__init__()
       self.config=config
       self.diffmap_model=CLIPEncoder(config.diffmap_config)
       self.protein_model=CLIPEncoder(config.protein_config) 
       self.diffmap_projection=ProjectionHead(input_dim=config.diffmap_config.hidden_size,output_dim=config.projection_dim,hidden_dim=config.projection_dim*2)
       self.protein_projection=ProjectionHead(input_dim=config.protein_config.hidden_size,output_dim=config.projection_dim,hidden_dim=config.projection_dim*2)
       self.logit_scale=nn.Parameter(torch.ones([])*config.logit_scale_init_value)
   def forward(self,diffmap_values,protein_values):
       diffmap_outputs=self.diffmap_model(diffmap_values)
       protein_outputs=self.protein_model(protein_values)
       diffmap_embeds=self.diffmap_projection(diffmap_outputs)
       protein_embeds=self.protein_projection(protein_outputs) 
       diffmap_embeds=F.normalize(diffmap_embeds,dim=-1)
       protein_embeds=F.normalize(protein_embeds,dim=-1)
       logit_scale=self.logit_scale.exp()
       logits_per_diffmap_protein=torch.matmul(diffmap_embeds,protein_embeds.t())*logit_scale
       return{"logits_per_diffmap_protein":logits_per_diffmap_protein,"diffmap_embeds":diffmap_embeds,"protein_embeds":protein_embeds}
class OptimizedCLIPModule(nn.Module):
   def __init__(self,config):
       super().__init__()
       self.config=config
       self.protein_embedding_cache=torch.zeros((config.cache_size,config.projection_dim),device="cuda" if torch.cuda.is_available() else "cpu")
       self.cache_ptr=0
       self.diffmap_model=CLIPEncoder(config.diffmap_config)
       self.protein_model=CLIPEncoder(config.protein_config)
       self.diffmap_projection=OptimizedProjectionHead(input_dim=config.diffmap_config.hidden_size,output_dim=config.projection_dim,hidden_dim=config.projection_dim*4)
       self.protein_projection=OptimizedProjectionHead(input_dim=config.protein_config.hidden_size,output_dim=config.projection_dim,hidden_dim=config.projection_dim*4)
       self.logit_scale=nn.Parameter(torch.ones([])*np.log(1/0.07))
   def update_cache(self,protein_embeds):
       batch_size=protein_embeds.size(0)
       if self.cache_ptr+batch_size>self.config.cache_size:self.cache_ptr=0
       self.protein_embedding_cache[self.cache_ptr:self.cache_ptr+batch_size]=protein_embeds.detach()
       self.cache_ptr=(self.cache_ptr+batch_size)%self.config.cache_size
   def forward(self,diffmap_values,protein_values,gather_distributed=True):
       diffmap_outputs=self.diffmap_model(diffmap_values)
       protein_outputs=self.protein_model(protein_values)
       diffmap_embeds=self.diffmap_projection(diffmap_outputs)
       protein_embeds=self.protein_projection(protein_outputs)
       diffmap_embeds=F.normalize(diffmap_embeds,dim=-1)
       protein_embeds=F.normalize(protein_embeds,dim=-1)
       self.update_cache(protein_embeds)
       logit_scale=self.logit_scale.exp().clamp(max=100)
       if gather_distributed and dist.is_initialized():
           world_size=dist.get_world_size()
           diffmap_embeds_gathered=[torch.zeros_like(diffmap_embeds)for _ in range(world_size)]
           protein_embeds_gathered=[torch.zeros_like(protein_embeds)for _ in range(world_size)]
           dist.all_gather(diffmap_embeds_gathered,diffmap_embeds)
           dist.all_gather(protein_embeds_gathered,protein_embeds)
           diffmap_embeds=torch.cat(diffmap_embeds_gathered,dim=0)
           protein_embeds=torch.cat(protein_embeds_gathered,dim=0)
       sim_d_p=torch.matmul(diffmap_embeds,protein_embeds.t())*logit_scale
       sim_d_cache=torch.matmul(diffmap_embeds,self.protein_embedding_cache[:self.cache_ptr].t())*logit_scale
       return{"logits_per_diffmap_protein":sim_d_p,"logits_per_diffmap_cache":sim_d_cache,"diffmap_embeds":diffmap_embeds,"protein_embeds":protein_embeds}
def optimized_clip_loss(outputs,temperature=0.07):
   sim_d_p=outputs["logits_per_diffmap_protein"]
   sim_d_cache=outputs["logits_per_diffmap_cache"]
   combined_sim=torch.cat([sim_d_p,sim_d_cache],dim=1)
   batch_size=sim_d_p.size(0)
   labels=torch.arange(batch_size).to(sim_d_p.device)
   smooth_factor=0.1
   n_categories=combined_sim.size(1)
   smooth_labels=torch.full_like(combined_sim,smooth_factor/(n_categories-1))
   smooth_labels.scatter_(1,labels.unsqueeze(1),1.0-smooth_factor)
   loss_d=F.cross_entropy(combined_sim,labels)
   loss_p=F.cross_entropy(sim_d_p.t(),labels)
   return(loss_d+loss_p)/2





class ImmuneCellDataset(Dataset):
   def __init__(self,adata,markers,transform=None):
       self.X_diffmap=torch.FloatTensor(adata.obsm['X_diffmap'])
       self.markers=torch.FloatTensor(markers)
       self.transform=transform
   def __len__(self):return len(self.X_diffmap)
   def __getitem__(self,idx):
       diffmap=self.X_diffmap[idx]
       marker=self.markers[idx]
       if self.transform:diffmap=self.transform(diffmap)
       return diffmap,marker
class GaussianNoise:
   def __init__(self,std=0.1):self.std=std
   def __call__(self,x):return x+torch.randn_like(x)*self.std
def train_epoch(model,train_loader,optimizer,device,config):
   model.train()
   total_loss=0
   correct=0
   total=0
   for diffmap_batch,protein_batch in train_loader:
       diffmap_batch,protein_batch=diffmap_batch.to(device),protein_batch.to(device)
       optimizer.zero_grad()
       outputs=model(diffmap_batch,protein_batch)
       if isinstance(model.module,OptimizedCLIPModule):
           loss=optimized_clip_loss(outputs)
       else:
           logits=outputs["logits_per_diffmap_protein"]
           loss=F.cross_entropy(logits,torch.arange(len(diffmap_batch)).to(device))
       loss.backward()
       nn.utils.clip_grad_norm_(model.parameters(),1.0)
       optimizer.step()
       total_loss+=loss.item()
       pred=logits.argmax(dim=1)
       correct+=(pred==torch.arange(len(diffmap_batch)).to(device)).sum().item()
       total+=len(diffmap_batch)
   return total_loss/len(train_loader),correct/total
def evaluate(model,val_loader,device):
   model.eval()
   correct=0
   total=0
   cosine_sims=[]
   with torch.no_grad():
       for diffmap_batch,protein_batch in val_loader:
           diffmap_batch,protein_batch=diffmap_batch.to(device),protein_batch.to(device)
           outputs=model(diffmap_batch,protein_batch)
           logits=outputs["logits_per_diffmap_protein"]
           pred=logits.argmax(dim=1)
           correct+=(pred==torch.arange(len(diffmap_batch)).to(device)).sum().item()
           total+=len(diffmap_batch)
           diffmap_embeds=outputs["diffmap_embeds"]
           protein_embeds=outputs["protein_embeds"]
           cosine_sim=F.cosine_similarity(diffmap_embeds.unsqueeze(1),protein_embeds.unsqueeze(0),dim=2)
           cosine_sims.append(cosine_sim.cpu())
   cosine_sims=torch.cat(cosine_sims,dim=0)
   return correct/total,cosine_sims
def run_experiment(config,train_dataset,val_dataset,test_datasets):
   device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
   train_loader=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
   val_loader=DataLoader(val_dataset,batch_size=config.batch_size)
   test_loaders={name:DataLoader(dataset,batch_size=config.batch_size)for name,dataset in test_datasets.items()}
   results={}
   for model_type in ['base','optimized']:
       if model_type=='base':
           model=DiffMapProteinCLIPModule(config)
       else:
           model=OptimizedCLIPModule(config)
       model=DDP(model.to(device))
       optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=0.01)
       scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs)
       best_val_acc=0
       for epoch in range(config.epochs):
           train_loss,train_acc=train_epoch(model,train_loader,optimizer,device,config)
           val_acc,val_cosine_sims=evaluate(model,val_loader,device)
           scheduler.step()
           if val_acc>best_val_acc:
               best_val_acc=val_acc
               test_results={}
               for dataset_name,test_loader in test_loaders.items():
                   test_acc,test_cosine_sims=evaluate(model,test_loader,device)
                   test_results[dataset_name]={'acc':test_acc,'cosine_sims':test_cosine_sims}
               results[model_type]={"val_acc":best_val_acc,"test_results":test_results}
   return results
def run_all_experiments():
   base_config=HybridCLIPConfig(
       diffmap_config=CLIPConfig(hidden_size=50,num_hidden_layers=2),
       protein_config=CLIPConfig(hidden_size=2000,num_hidden_layers=2),
       projection_dim=128,
       logit_scale_init_value=np.log(1/0.07),
       cache_size=8192,
       batch_size=128,
       learning_rate=3e-4,
       epochs=100
   )
   configs={
       'batch_size':[32,64,128,256],
       'embed_dim':[32,64,128,256,512],
       'temperature':[0.05,0.07,0.1],
       'architecture':['mlp','transformer3','transformer6','resnet']
   }
   all_results={}
   for param,values in configs.items():
       param_results={}
       for value in values:
           config=copy.deepcopy(base_config)
           setattr(config,param,value)
           results=run_experiment(config,train_dataset,val_dataset,test_datasets)
           param_results[value]=results
       all_results[param]=param_results
   return all_results






class CLIPConfig:
   def __init__(self,hidden_size,num_hidden_layers):
       self.hidden_size=hidden_size
       self.num_hidden_layers=num_hidden_layers
       self.layer_norm_eps=1e-5
class HybridCLIPConfig:
   def __init__(self,diffmap_config,protein_config,projection_dim,logit_scale_init_value,cache_size,batch_size,learning_rate,epochs):
       self.diffmap_config=diffmap_config
       self.protein_config=protein_config
       self.projection_dim=projection_dim
       self.logit_scale_init_value=logit_scale_init_value
       self.cache_size=cache_size
       self.batch_size=batch_size
       self.learning_rate=learning_rate
       self.epochs=epochs
def load_datasets():
   immune=sc.read('TS_immune.h5ad')
   k562=sc.read('norman_k562.h5ad')
   sc.pp.normalize_total(immune)
   sc.pp.normalize_total(k562)
   sc.pp.log1p(immune)
   sc.pp.log1p(k562)
   immune_markers=pd.read_csv('immune_markers.csv')
   k562_markers=pd.read_csv('k562_markers.csv')
   train_idx,val_idx=train_test_split(range(len(immune)),test_size=0.15)
   train_dataset=ImmuneCellDataset(immune[train_idx],immune_markers.iloc[train_idx])
   val_dataset=ImmuneCellDataset(immune[val_idx],immune_markers.iloc[val_idx])
   test_datasets={'ImmGen':ImmuneCellDataset(sc.read('immgen.h5ad'),pd.read_csv('immgen_markers.csv')),
                 'HCA':ImmuneCellDataset(sc.read('hca.h5ad'),pd.read_csv('hca_markers.csv')),
                 'CITE-seq':ImmuneCellDataset(sc.read('cite_seq.h5ad'),pd.read_csv('cite_seq_markers.csv')),
                 'K562':ImmuneCellDataset(k562,k562_markers)}
   return train_dataset,val_dataset,test_datasets
def analyze_results(all_results):
   analysis={}
   for param,param_results in all_results.items():
       param_analysis={'base_acc':[],'opt_acc':[],'base_cosine':[],'opt_cosine':[]}
       for value,results in param_results.items():
           param_analysis['base_acc'].append(results['base']['val_acc'])
           param_analysis['opt_acc'].append(results['optimized']['val_acc'])
           param_analysis['base_cosine'].append(results['base']['test_results']['ImmGen']['cosine_sims'].mean().item())
           param_analysis['opt_cosine'].append(results['optimized']['test_results']['ImmGen']['cosine_sims'].mean().item())
       analysis[param]=param_analysis
   return analysis
def compute_confusion_matrix(model,loader,device,num_classes):
   confusion=torch.zeros(num_classes,num_classes)
   model.eval()
   with torch.no_grad():
       for diffmap_batch,protein_batch in loader:
           diffmap_batch,protein_batch=diffmap_batch.to(device),protein_batch.to(device)
           outputs=model(diffmap_batch,protein_batch)
           logits=outputs["logits_per_diffmap_protein"]
           pred=logits.argmax(dim=1)
           for t,p in zip(torch.arange(len(diffmap_batch)),pred):
               confusion[t.long(),p.long()]+=1
   return confusion
def main():
   torch.manual_seed(42)
   np.random.seed(42)
   train_dataset,val_dataset,test_datasets=load_datasets()
   base_config=HybridCLIPConfig(
       diffmap_config=CLIPConfig(hidden_size=50,num_hidden_layers=2),
       protein_config=CLIPConfig(hidden_size=2000,num_hidden_layers=2),
       projection_dim=128,
       logit_scale_init_value=np.log(1/0.07),
       cache_size=8192,
       batch_size=128,
       learning_rate=3e-4,
       epochs=100
   )
   dist.init_process_group(backend='nccl')
   all_results=run_all_experiments()
   analysis=analyze_results(all_results)
   torch.save({'config':base_config,'results':all_results,'analysis':analysis},'clip_experiments.pt')
   dist.destroy_process_group()
if __name__=="__main__":
   main()







def analyze_cell_type_confusion(confusion_matrix,cell_types):
   confusion_rates={}
   t_cell_pairs=[('cd4+','cd8+'),('naive','memory'),('regulatory','helper')]
   myeloid_pairs=[('monocyte','macrophage'),('dc_subtype1','dc_subtype2')]
   for pair in t_cell_pairs+myeloid_pairs:
       idx1,idx2=cell_types.index(pair[0]),cell_types.index(pair[1])
       confusion=confusion_matrix[idx1,idx2]+confusion_matrix[idx2,idx1]
       total=confusion_matrix[idx1].sum()+confusion_matrix[idx2].sum()
       confusion_rates[f"{pair[0]}_vs_{pair[1]}"]=confusion/total*100
   return confusion_rates
def analyze_embedding_collapse(diffmap_embeds,protein_embeds,cell_types,cell_type_groups):
   similarities={}
   for group,types in cell_type_groups.items():
       indices=[cell_types.index(t)for t in types]
       group_diffmap=diffmap_embeds[indices]
       group_protein=protein_embeds[indices]
       sim_matrix=F.cosine_similarity(group_diffmap.unsqueeze(1),group_diffmap.unsqueeze(0),dim=2)
       similarities[group]=sim_matrix.mean().item()
   return similarities
def detailed_evaluation(model,loader,cell_types,cell_type_groups,device):
   model.eval()
   all_diffmap_embeds,all_protein_embeds=[],[]
   with torch.no_grad():
       for diffmap_batch,protein_batch in loader:
           diffmap_batch,protein_batch=diffmap_batch.to(device),protein_batch.to(device)
           outputs=model(diffmap_batch,protein_batch)
           all_diffmap_embeds.append(outputs["diffmap_embeds"])
           all_protein_embeds.append(outputs["protein_embeds"])
   all_diffmap_embeds=torch.cat(all_diffmap_embeds,dim=0)
   all_protein_embeds=torch.cat(all_protein_embeds,dim=0)
   confusion_matrix=compute_confusion_matrix(model,loader,device,len(cell_types))
   confusion_rates=analyze_cell_type_confusion(confusion_matrix,cell_types)
   embedding_collapse=analyze_embedding_collapse(all_diffmap_embeds,all_protein_embeds,cell_types,cell_type_groups)
   return{"confusion_rates":confusion_rates,"embedding_collapse":embedding_collapse}
def visualize_results(analysis):
   base_accs,opt_accs={},{}
   for param,results in analysis.items():
       base_accs[param]=np.array(results['base_acc'])
       opt_accs[param]=np.array(results['opt_acc'])
   batch_size_results=pd.DataFrame({'batch_size':[32,64,128,256],'base':base_accs['batch_size'],'optimized':opt_accs['batch_size']})
   embed_dim_results=pd.DataFrame({'dim':[32,64,128,256,512],'base':base_accs['embed_dim'],'optimized':opt_accs['embed_dim']})
   temperature_results=pd.DataFrame({'temp':[0.05,0.07,0.1],'base':base_accs['temperature'],'optimized':opt_accs['temperature']})
   return batch_size_results,embed_dim_results,temperature_results
def run_exhaustive_experiments():
   cell_type_groups={'t_cell':['cd4+','cd8+','naive','memory','regulatory','helper'],
                    'b_cell':['naive_b','memory_b','plasma'],
                    'myeloid':['monocyte','macrophage','dc_subtype1','dc_subtype2']}
   architectures={'mlp':{'layers':2,'type':'mlp'},'transformer3':{'layers':3,'type':'transformer'},
                 'transformer6':{'layers':6,'type':'transformer'},'resnet':{'layers':4,'type':'resnet'}}
   augmentations=[None,GaussianNoise(0.1),lambda x:F.dropout(x,p=0.2),lambda x:x*torch.bernoulli(torch.ones_like(x)*0.8)]
   loss_variants=['standard','label_smoothing','hard_negative','supervised_contrastive']
   all_results={}
   for arch_name,arch_config in architectures.items():
       for aug in augmentations:
           for loss in loss_variants:
               config=copy.deepcopy(base_config)
               config.architecture=arch_config
               results=run_experiment(config,train_dataset,val_dataset,test_datasets,augmentation=aug,loss_type=loss)
               all_results[f"{arch_name}_{aug}_{loss}"]=results
   return all_results
def compute_marker_space_analysis(protein_embeds,marker_pairs):
   similarities={}
   for pair_name,(marker1,marker2)in marker_pairs.items():
       idx1,idx2=marker_names.index(marker1),marker_names.index(marker2)
       sim=F.cosine_similarity(protein_embeds[idx1:idx1+1],protein_embeds[idx2:idx2+1]).item()
       similarities[pair_name]=sim
   return similarities
if __name__=="__main__":
   marker_pairs={'hierarchical':('CD3','CD19'),'subtype':('CD4','CD8')}
   cell_type_groups={'t_cell':['cd4+','cd8+','naive','memory','regulatory','helper'],
                    'b_cell':['naive_b','memory_b','plasma'],
                    'myeloid':['monocyte','macrophage','dc_subtype1','dc_subtype2']}
   train_dataset,val_dataset,test_datasets=load_datasets()
   base_config=HybridCLIPConfig(diffmap_config=CLIPConfig(hidden_size=50,num_hidden_layers=2),
                               protein_config=CLIPConfig(hidden_size=2000,num_hidden_layers=2),
                               projection_dim=128,logit_scale_init_value=np.log(1/0.07),
                               cache_size=8192,batch_size=128,learning_rate=3e-4,epochs=100)
   dist.init_process_group(backend='nccl')
   results=run_exhaustive_experiments()
   detailed_metrics=detailed_evaluation(model,val_loader,cell_types,cell_type_groups,device)
   marker_analysis=compute_marker_space_analysis(protein_embeds,marker_pairs)
   analysis=analyze_results(results)
   batch_results,dim_results,temp_results=visualize_results(analysis)
   torch.save({'config':base_config,'results':results,'analysis':analysis,
              'detailed_metrics':detailed_metrics,'marker_analysis':marker_analysis,
              'batch_results':batch_results,'dim_results':dim_results,
              'temp_results':temp_results},'clip_exhaustive_results.pt')
   dist.destroy_process_group()






def analyze_embedding_distributions(diffmap_embeds,protein_embeds,cell_types,num_components=3):
   pca=PCA(n_components=num_components)
   diffmap_pca=pca.fit_transform(diffmap_embeds.cpu().numpy())
   protein_pca=pca.fit_transform(protein_embeds.cpu().numpy())
   distributions={'diffmap':{},'protein':{}}
   for i,cell_type in enumerate(cell_types):
       mask=cell_labels==i
       distributions['diffmap'][cell_type]={'mean':diffmap_pca[mask].mean(0),'std':diffmap_pca[mask].std(0)}
       distributions['protein'][cell_type]={'mean':protein_pca[mask].mean(0),'std':protein_pca[mask].std(0)}
   return distributions
def track_training_dynamics(model,train_loader,val_loader,optimizer,device,epochs):
   metrics={'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[],'embedding_stats':[]}
   for epoch in range(epochs):
       train_loss,train_acc=train_epoch(model,train_loader,optimizer,device)
       val_loss,val_acc,embeds=evaluate_with_embeddings(model,val_loader,device)
       embed_stats={'diffmap_norm':torch.norm(embeds['diffmap'],dim=1).mean().item(),
                   'protein_norm':torch.norm(embeds['protein'],dim=1).mean().item(),
                   'similarity_stats':analyze_similarity_distribution(embeds['diffmap'],embeds['protein'])}
       metrics['train_loss'].append(train_loss)
       metrics['train_acc'].append(train_acc)
       metrics['val_loss'].append(val_loss)
       metrics['val_acc'].append(val_acc)
       metrics['embedding_stats'].append(embed_stats)
   return metrics
def analyze_failure_cases(model,loader,cell_types,device):
   failures=defaultdict(list)
   model.eval()
   with torch.no_grad():
       for diffmap_batch,protein_batch,labels in loader:
           diffmap_batch,protein_batch=diffmap_batch.to(device),protein_batch.to(device)
           outputs=model(diffmap_batch,protein_batch)
           logits=outputs["logits_per_diffmap_protein"]
           pred=logits.argmax(dim=1)
           for i,(p,t)in enumerate(zip(pred,labels)):
               if p!=t:
                   failures[f"{cell_types[t.item()]}->{cell_types[p.item()]}"].append({
                       'diffmap_embed':outputs["diffmap_embeds"][i].cpu(),
                       'protein_embed':outputs["protein_embeds"][i].cpu(),
                       'confidence':logits[i].softmax(0)[p].item()})
   return failures
def cross_dataset_analysis(model,test_loaders,device):
   results={}
   for dataset_name,loader in test_loaders.items():
       acc,cosine_sims=evaluate(model,loader,device)
       confusion=compute_confusion_matrix(model,loader,device,len(cell_types))
       error_types=analyze_error_patterns(confusion,cell_types)
       embedding_stats=compute_embedding_statistics(model,loader,device)
       results[dataset_name]={'accuracy':acc,'cosine_similarities':cosine_sims,
                            'error_types':error_types,'embedding_stats':embedding_stats}
   return results
def compare_architectures_detailed(results):
   architecture_comparison={'base_mlp':{},'transformer3':{},'transformer6':{},'resnet':{}}
   metrics=['accuracy','loss','embedding_quality','generalization']
   for arch in architecture_comparison:
       for metric in metrics:
           values=extract_metric_for_architecture(results,arch,metric)
           architecture_comparison[arch][metric]={'mean':np.mean(values),'std':np.std(values)}
   return architecture_comparison
def analyze_hard_negatives_impact(model,loader,device):
   cache_stats={'hit_rate':[],'negative_difficulty':[],'gradient_magnitude':[]}
   model.train()
   for diffmap_batch,protein_batch in loader:
       diffmap_batch,protein_batch=diffmap_batch.to(device),protein_batch.to(device)
       outputs=model(diffmap_batch,protein_batch)
       cache_hits=compute_cache_statistics(outputs["logits_per_diffmap_cache"])
       negative_difficulty=measure_negative_difficulty(outputs)
       grad_magnitude=compute_gradient_statistics(model)
       cache_stats['hit_rate'].append(cache_hits)
       cache_stats['negative_difficulty'].append(negative_difficulty)
       cache_stats['gradient_magnitude'].append(grad_magnitude)
   return cache_stats
def run_comprehensive_experiments():
   base_results=run_all_experiments()
   training_dynamics=track_training_dynamics(model,train_loader,val_loader,optimizer,device,config.epochs)
   cross_dataset_results=cross_dataset_analysis(model,test_loaders,device)
   failure_analysis=analyze_failure_cases(model,val_loader,cell_types,device)
   architecture_comparison=compare_architectures_detailed(base_results)
   hard_negative_stats=analyze_hard_negatives_impact(model,train_loader,device)
   embedding_distributions=analyze_embedding_distributions(diffmap_embeds,protein_embeds,cell_types)
   detailed_metrics=detailed_evaluation(model,val_loader,cell_types,cell_type_groups,device)
   marker_space=compute_marker_space_analysis(protein_embeds,marker_pairs)
   augmentation_impact=analyze_augmentation_impact(results)
   loss_variant_comparison=compare_loss_variants(results)
   additional_metrics={'training_dynamics':training_dynamics,
                      'cross_dataset':cross_dataset_results,
                      'failures':failure_analysis,
                      'architecture_comparison':architecture_comparison,
                      'hard_negatives':hard_negative_stats,
                      'embedding_distributions':embedding_distributions,
                      'detailed_metrics':detailed_metrics,
                      'marker_space':marker_space,
                      'augmentation_impact':augmentation_impact,
                      'loss_comparison':loss_variant_comparison}
   return additional_metrics
if __name__=="__main__":
   comprehensive_results=run_comprehensive_experiments()
   torch.save({'base_results':results,'comprehensive_results':comprehensive_results},'final_clip_analysis.pt')





