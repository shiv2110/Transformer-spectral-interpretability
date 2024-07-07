import numpy as np
import torch
import copy
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eig

import torch.nn.functional as F
from torch.nn import Linear

from pymatting.util.util import row_sum
from scipy.sparse import diags
from scipy.stats import skew
# from .eigenshuffle import eigenshuffle
# from sentence_transformers import SentenceTransformer
# from torch.nn import CosineSimilarity as CosSim

from spectral.get_fev import get_eigs, get_grad_eigs, avg_heads


class GeneratorOurs:
    def __init__(self, model_usage, save_visualization=False):
        self.model_usage = model_usage
        # self.save_visualization = save_visualization


    def generate_ours_dsm(self, input, how_many = 5, index=None):


        output = self.model_usage.forward(input).question_answering_score
        # print(f"{output.last_hidden_state.shape}")
        model = self.model_usage.model

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(model.device) * output) #baka
        
        # one_hot = torch.sum(one_hot.cuda() * output) #baka

        model.zero_grad()
        one_hot.backward(retain_graph=True)



        
        image_fev = get_eigs(model.lxmert.encoder.visual_feats_list_x[-2], 
                                                 "image", how_many)
        
        lang_fev = get_eigs(model.lxmert.encoder.lang_feats_list_x[-1], 
                                               "text", how_many)

        return [lang_fev], [image_fev]



    def generate_ours_dsm_grad(self, input, how_many = 5, index = None):

        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(model.device) * output) #baka

        # one_hot = torch.sum(one_hot.cuda() * output) #baka

        model.zero_grad()
        one_hot.backward(retain_graph=True)

        image_flen = len(model.lxmert.encoder.visual_feats_list_x)
        text_flen = len(model.lxmert.encoder.lang_feats_list_x)

        blk = model.lxmert.encoder.x_layers

        def get_layer_wise_fevs (feats_list, flen, modality, how_many):
            # blk_count = 0
            layer_wise_fevs = []
            
            for i in range(flen):
                if modality == "image":
                    grad = blk[i].visn_self_att.self.get_attn_gradients().detach()
                else:
                    grad = blk[i].lang_self_att.self.get_attn_gradients().detach()

                # blk_count += 1
                fev = get_grad_eigs(feats_list[i], modality, grad, model.device, how_many)
                layer_wise_fevs.append( fev )
      
            return layer_wise_fevs

        
        image_fevs = get_layer_wise_fevs(model.lxmert.encoder.visual_feats_list_x, 
                                                 image_flen - 1, "image", how_many)
        
        lang_fevs = get_layer_wise_fevs(model.lxmert.encoder.lang_feats_list_x, 
                                               text_flen, "text", how_many)


        image_fev = torch.stack(image_fevs, dim=0).sum(dim=0)
        lang_fev = torch.stack(lang_fevs, dim=0).sum(dim=0)
        # new_fev1 = (new_fev1 - torch.min(new_fev1))/(torch.max(new_fev1) - torch.min(new_fev1))
        # new_fev = (new_fev - torch.min(new_fev))/(torch.max(new_fev) - torch.min(new_fev))

        return [lang_fev], [image_fev]
    

    def generate_ours_dsm_grad_cam(self, input, how_many = 5, index = None):
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(model.device) * output) #baka
        # one_hot = torch.sum(one_hot.cuda() * output) #baka
        model.zero_grad()
        one_hot.backward(retain_graph=True)


        image_flen1 = len(model.lxmert.encoder.visual_feats_list_x)
        text_flen1 = len(model.lxmert.encoder.lang_feats_list_x)

        def get_layer_wise_fevs1 (feats_list, flen, modality, how_many):
            # blk_count = 0
            layer_wise_fevs = []
            blk = model.lxmert.encoder.x_layers
            # lang_blk = model.lxmert.encoder.layer

            for i in range(flen):
                # feats = F.normalize(feats_list[i].detach().clone().squeeze().cpu(), p = 2, dim = -1)
                # print(f"Features' shape: {feats.shape}")
                fev = get_eigs(feats_list[i], modality, how_many)
                if modality == "text":
                    fev = fev[1:-1]
                    # fev = torch.cat( ( torch.zeros(1), fev ) )

                # layer_wise_fevs.append( eigenvalues[fev_idx].real * fev )
                if modality == "image":
                    grad = blk[i].visn_self_att.self.get_attn_gradients().detach()
                    cam = blk[i].visn_self_att.self.get_attn().detach()
                    cam = avg_heads(cam, grad)

                    fev = fev.to(model.device)
                    fev = cam @ fev.unsqueeze(1)
                    fev = fev[:, 0]

                else:
                    grad = blk[i].lang_self_att.self.get_attn_gradients().detach()[:, :, 1:-1, 1:-1]
                    cam = blk[i].lang_self_att.self.get_attn().detach()[:, :, 1:-1, 1:-1]
                    cam = avg_heads(cam, grad)

                    fev = fev.to(model.device)
                    fev = cam @ fev.unsqueeze(1)
                    fev = fev[:, 0]
                    fev = torch.cat( ( torch.zeros(1).to(model.device), fev, torch.zeros(1).to(model.device)  ) )


                layer_wise_fevs.append( torch.abs(fev) )
      
            return layer_wise_fevs


        image_fevs = get_layer_wise_fevs1(model.lxmert.encoder.visual_feats_list_x, 
                                                 image_flen1 - 1, "image", how_many)
        
        lang_fevs = get_layer_wise_fevs1(model.lxmert.encoder.lang_feats_list_x, 
                                               text_flen1, "text", how_many)


        # return lang_fevs[-2], image_fevs[-2], eigenvalues_image, eigenvalues_text
        new_fev_image = torch.stack(image_fevs, dim=0).sum(dim=0)
        new_fev_lang = torch.stack(lang_fevs, dim=0).sum(dim=0)
        # new_fev1 = (new_fev1 - torch.min(new_fev1))/(torch.max(new_fev1) - torch.min(new_fev1))
        # new_fev = (new_fev - torch.min(new_fev))/(torch.max(new_fev) - torch.min(new_fev))

        return [new_fev_lang], [new_fev_image]





 