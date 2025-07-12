import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

import math
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .clip_text import clip as clipO

_tokenizer = _Tokenizer()
##############
def matrixdistance_image(token,feature,lambda_factor=0.8):
    cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
    
    token=token.float()
    feature=feature.float()
    token = token / token.norm(dim=1, keepdim=True) 
    feature = feature / feature.norm(dim=1, keepdim=True)
                            
    abs_diff = torch.abs(token - feature)
    
    # 计算平均绝对误差 (MAE)
    mae = torch.mean(abs_diff)
    
    # 获取张量的最小值和最大值
    min_value = torch.min(torch.cat((token, feature)))
    max_value = torch.max(torch.cat((token, feature)))
    
    # 归一化 MAE
    score = mae / (max_value - min_value)
    

    

    return score
def adaptFeature(matrix_entro,target):    # target is token , 用于合并计算

    # matrix_entro = matrix_entro / matrix_entro.norm(dim=1, keepdim=True)
    matrix_entro = matrix_entro / matrix_entro.norm(dim=1, keepdim=True) 
    target = target / target.norm(dim=1, keepdim=True)    


    # 将两个归一化后的张量相加
    sum_norm = matrix_entro + target

    # 恢复到归一化前的尺度
    # restored_sum = sum_norm * target.norm(dim=1, keepdim=True)

    #result = (1/target.shape[1])*restored_sum

    return sum_norm # result

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

def construct_prompts( ctx,ctx_ind, prefix, suffix,label=None,):
    # dim0 is either batch_size (during training) or n_cls (during testing)
    # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
    # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
    # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

    if label is not None:
        prefix = prefix[label]
        suffix = suffix[label]
    
    #ctx=self.calentropY(self.matrix_entropies,ctx)
    ctx=torch.cat((ctx_ind,ctx),dim=1)
        # if ctx_ind.dim() == 2:
        #     ctx_ind = ctx_ind.unsqueeze(0).expand(self.n_cls, -1, -1)

    prompts = torch.cat(
        [
            prefix,  # (dim0, 1, dim)
            ctx,  # (dim0, n_ctx, dim)
            suffix,  # (dim0, *, dim)
        ],
        dim=1,
    )

    return prompts
    
     
def entropyx(signal):
    prob_dist = torch.abs(signal) / torch.sum(torch.abs(signal))


    entropy = -torch.sum(prob_dist * torch.log2(prob_dist + 1e-10))
    return entropy

def calentrop(matrix): #测量输入的矩阵的熵
    freq_space=[]
    for item in range(matrix.shape[0]): # 50 * 77 * 512
        fft_result = torch.fft.fft2(matrix[item,:,:].to(torch.float32))
        freq_space.append(fft_result)
    freq_spaces=torch.stack(freq_space,dim=0)
 
    matrix_entropies = []
    for item_freq in freq_spaces:  
        # item_freq 77* 512
        item_entropies=[]
        for i in range(item_freq.shape[1]):
            column_entropy = entropyx(item_freq[ :, i])
            item_entropies.append(column_entropy)
        # matrix_entro=torch.stack(item_entropies,dim=0)/entropyx(item_freq)

        matrix_entro=torch.stack(item_entropies,dim=0)/entropyx(item_freq)
        matrix_entropies.append(matrix_entro)
    stacked_matrix = torch.stack(matrix_entropies, dim=0) # 将 category 数目堆叠
    # average_vector = torch.mean(stacked_matrix, dim=0, keepdim=True)
    return stacked_matrix # 50*512


# def calentropY(matrix_entro,target):    # target is token , 用于合并计算
#     target_new=[]
#     ffti=[]
#     for singleTokens in range(target.shape[0]):
#             ffti_item = torch.fft.fft2(target[singleTokens,:,:].to(torch.float32))
#             ffti.append(ffti_item)
#     fftis=torch.stack(ffti, dim=0)

#     for i in range(fftis.shape[0]): #ffti 50*2*512  ,matrix_entro 50*512

#         result = (1/target.shape[1])*matrix_entro[i].unsqueeze(0).to(target) * fftis[i].to(target)
        
#         ifft_result = torch.fft.ifft2(result.float())
#         target_new.append(ifft_result)  
#     target_n=torch.stack(target_new,dim=0)
#     return target_n

def calentropY(matrix_entro,target):    # target is token , 用于合并计算

    ffti=[]
    for singleTokens in range(target.shape[0]):
            ffti_item = torch.fft.fft2(target[singleTokens,:,:].to(torch.float32))
            ffti.append(ffti_item)
    fftis=torch.stack(ffti, dim=0)

     #ffti 50*2*512  ,matrix_entro 50*512

    result = (1/target.shape[1])*matrix_entro.unsqueeze(1).to(target) * fftis.to(target)
    
    final=[]
    for singleTokens in range(result.shape[0]):
            ffti_item = torch.fft.ifft2(result[singleTokens,:,:].to(torch.float32))
            final.append(ffti_item)
    final=torch.stack(final, dim=0)
    return final


def fftshift(tensor):
    """Shift zero frequency component to the center of the spectrum."""
    ndim = len(tensor.shape)
    for dim in range(ndim):
        tensor = torch.roll(tensor, tensor.shape[dim] // 2, dim)
    return tensor

def gaussian_filter(size, sigma):
    """Generate a Gaussian filter in frequency domain."""
    rows = torch.fft.fftfreq(size[0]).reshape(-1, 1)
    cols = torch.fft.fftfreq(size[1]).reshape(1, -1)
    radius = torch.sqrt(rows**2 + cols**2)
    filter = torch.exp(-radius**2 / (2 * sigma**2))
    return filter

def filter_frequency(im, sigma):
    # Apply FFT and shift the zero frequency component to the center
    lists=[]
    for index,i in enumerate(im):
        f = torch.fft.fft2(i)
        fshift = fftshift(f).to(i)

        # Apply Gaussian filter
        gaussian = gaussian_filter(i.shape[-2:], sigma).to(i)
        fshift *= gaussian

        # Inverse FFT to get the image back
        f_ishift = fftshift(fshift)
        i = torch.fft.ifft2(f_ishift.to(i)).real
        lists.append(i)
    
    stacked_img=torch.stack(lists,dim=0)
    
    return stacked_img  

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def load_clip_to_cpuO(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clipO._MODELS[backbone_name]
    model_path = clipO._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    with torch.no_grad():
        model = clipO.build_model(state_dict or model.state_dict())

    return model




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text,embeddingx):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0,embeddingx]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        clone_feature = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)].clone()
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x,clone_feature


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_T = 2 # cfg.TRAINER.MAPLE.N_CTX
        n_ctx_zero=2
        n_ctx=n_ctx_T+n_ctx_zero
        
        
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.dtype = clip_model.dtype
        
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        clip_model_ = load_clip_to_cpuO(cfg)
        clip_model_.cuda()
        
        init_style=True
        if init_style==True:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0, 1: 1 + n_ctx_T, :]
            ctx_vectors_zero = embedding[0, 1: 1 + n_ctx_zero, :]
            prompt_prefix = ctx_init
            
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx_T, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            ctx_vectors_zero = torch.empty(n_ctx_zero, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_zero, std=0.02)
            prompt_prefix = " ".join(["X"] * 4)
            
            
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj_zero = nn.Linear(ctx_dim, 768)
        

        self.ctx = nn.Parameter(ctx_vectors_zero.clone())
        self.proj_zero=self.proj_zero.to(torch.float32)
        
        
        
        self.ctx_ind= nn.Parameter(ctx_vectors.clone())
        self.proj=self.proj.to(torch.float32)
            
        # coeff = nn.Parameter(torch.tensor(0.5))
        # self.coeff = coeff
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx_T, 512)).to(torch.float32)
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        self.compound_prompts_fuxi_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx_zero, 512)).to(torch.float32)
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_fuxi_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt

        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer.to(torch.float32), self.compound_prompts_depth - 1)
        self.compound_prompt_projections_zero = _get_clones(single_layer.to(torch.float32), self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            
            text_features_,self.clone_feature_text = clip_model_.encode_text(tokenized_prompts.cuda())
            self.text_features_ = text_features_ / text_features_.norm(dim=-1, keepdim=True)
            self.clone_feature_text = self.clone_feature_text  / self.clone_feature_text .norm(dim=-1, keepdim=True)
            visual_encoder_ = clip_model_.visual
        for name, param in visual_encoder_.named_parameters():
                    param.requires_grad_(False)
        del clip_model_
        self.visual_encoder_ = visual_encoder_


        self.meta_net = nn.Sequential(
            OrderedDict([("linear1", nn.Linear(512, 128,bias=True))
                # ("linear1", nn.Linear(vis_dim, vis_dim // 4,bias=True)),
                         ,("relu", QuickGELU()),
                         ("linear2", nn.Linear(128, n_ctx_zero*512,bias=True))
                         ])).cuda()
        self.meta_net_ = nn.Sequential(
            OrderedDict([("linear1", nn.Linear(512, 64,bias=True))
                # ("linear1", nn.Linear(vis_dim, vis_dim // 4,bias=True)),
                         ,("relu", QuickGELU()),
                         ("linear2", nn.Linear(64, n_ctx_zero*512,bias=True))
                         ])).to(self.dtype).cuda()
        
        self.mini_net_ = nn.Sequential(
            OrderedDict([("linear1", nn.Linear(512, 512,bias=True))
                         ,("relu", QuickGELU()),
                         ("linear2", nn.Linear(512, 768,bias=True))
                         ])).cuda()        

        # self.mini_net = nn.Sequential(
        #     OrderedDict([("linear1", nn.Linear(768, 32,bias=True))
        #                  ,("relu", QuickGELU()),
        #                  ("linear2", nn.Linear(32, 512,bias=True))
        #                  ])).cuda()   
        
        # self.embeddingx = embedding
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        



        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # self.calentrop = calentrop
        # self.calentropY = calentropY
        # self.matrix_entropies=self.calentrop(self.embeddingx.to(self.ctx.device))

    def construct_prompts(self, ctx, prefix, suffix,label=None,):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        #ctx=self.calentropY(self.matrix_entropies,ctx)
        
    
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        
        try:
            text_features_ = self.meta_net(self.text_features_.to(torch.float32).to("cuda:0"))
        except:
            text_features_ = self.meta_net(self.text_features_.to(torch.float32).to("cuda:1"))
        text_features_ = text_features_.reshape(text_features_.shape[0],-1,512)
        
        ctx = self.ctx
        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        ctx_ind=self.ctx_ind
        if ctx_ind.dim() == 2:
            ctx_ind = ctx_ind.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        



        deep_compound_prompts_vision = []
        for index, layer in enumerate(self.compound_prompt_projections):
            deep_compound_prompts_vision.append(layer(self.compound_prompts_text[index]))
        self.ctx=self.ctx.cuda()
        self.ctx_ind=self.ctx_ind.cuda()

        deep_compound_prompts_fuxi_vision = []
        for index, layer in enumerate(self.compound_prompt_projections_zero):
            deep_compound_prompts_fuxi_vision.append(layer(self.compound_prompts_fuxi_text[index]))


        #return prompts, share_ctx, self.compound_prompts_text, deep_compound_prompts_vision,prompts,text_features_,self.meta_net_
        return [prefix,suffix,ctx],self.compound_prompts_text,deep_compound_prompts_vision,text_features_,self.meta_net_,self.proj,self.text_features_,self.visual_encoder_,ctx_ind,self.ctx_ind, self.compound_prompt_projections,self.clone_feature_text,self.compound_prompts_fuxi_text,deep_compound_prompts_fuxi_vision,self.mini_net_,self.proj_zero,self.compound_prompt_projections_zero



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model).cuda()
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        # self.image_encoder_ = clip_model.visualO # clip_model_.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        

        
        n_ctx = cfg.TRAINER.MAPLE.N_CTX



    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        self.prompt_learner=self.prompt_learner
        prompts_cobo, deep_compound_prompts_text, deep_compound_prompts_vision,\
            text_features_,meta_net_,proj,text_features_a,visual_encoder_ ,\
                ctx_ind,ctx_ind2d,compound_prompt_projections,clone_feature_text,\
                compound_prompts_fuxi_text,deep_compound_prompts_fuxi_vision,mini_net_,proj_zero,compound_prompt_projections_zero= self.prompt_learner()
        
        ######################################################################################################
        with torch.no_grad():    
                image_features_,upper_image=visual_encoder_(image.type(self.dtype))
                image_features_ = image_features_ / image_features_.norm(dim=-1, keepdim=True)
                upper_image = upper_image / upper_image.norm(dim=-1, keepdim=True)
                
        logits_ = logit_scale * image_features_ @ text_features_a.to(image_features_).t()
        max_values, max_indices = torch.max(logits_, dim=1)
        max_values, max_indices_top3 = logits_.topk(1, dim=1)
        
        condition_feature_ = meta_net_(image_features_.cuda())
        condition_feature_ = condition_feature_.reshape(condition_feature_.shape[0],-1,512) #2,2,512
        

######################################################################        
        prefix=prompts_cobo[0]
        suffix=prompts_cobo[1]
        ctx=prompts_cobo[2]
        
        
        feature_=[]
        prompt=[]
        share_ctx_=[]
        feature_Image=[]
        for i in range(len(image)):
 
                     
            image_f = condition_feature_[i]
            image_f = image_f / image_f.norm(dim=1, keepdim=True)
            
            
            text_features_ = text_features_ / text_features_.norm(dim=1, keepdim=True)       

            
            
            ## 初始化 prompt token construction
            text_features_[max_indices_top3[i]]=image_f + text_features_[max_indices_top3[i]]     
            feature_ii=(text_features_).type(self.dtype)
            # 合并图像，文本特征与 训练token
            feature_i=adaptFeature(feature_ii.to(ctx.device),ctx).type(self.dtype)        

            # 生成用于 text encoder 的prompt实例， 合并twins token
            prompts_i = construct_prompts(feature_i,ctx_ind, prefix, suffix).type(self.dtype)# text prompt 初始化
            prompt.append(prompts_i) 
                     
            # 生成用于 image encoder 的prompt实例， 合并twins token
            # feature_x=        feature_i[max_indices[i]].to(torch.float32) #找出特定图形特征
            # share_ctx_zero=   proj_zero(feature_x)      #转换   joint token
            
            
            
            fuxi_meta=feature_ii[max_indices[i]].to(torch.float32)
            fuxi_meta_proj=proj_zero(fuxi_meta)
            image_fuxi_proj_ctx=proj_zero(ctx.to(fuxi_meta))
            share_ctx_zero=adaptFeature(fuxi_meta_proj,image_fuxi_proj_ctx).type(self.dtype)
            
            
            share_ctx=proj(ctx_ind2d.to(torch.float32))  #转换 twins token
            share_ctx_.append(torch.cat((share_ctx,share_ctx_zero),dim=0)) 
            
            
            # 复习
            
            
            feature_.append([feature_ii,compound_prompts_fuxi_text]) # text fuxi          
            feature_Images = fuxi_meta_proj.unsqueeze(0) 
            feature_Image.append([feature_Images,deep_compound_prompts_fuxi_vision])#  image 复习

        
     
        
        
       
        
        logits = []
        clone_feature_images= []
        image_features_list=[]
        upper_texts=[]
        # for prompts_i, feature_i,share_ctxs,feature_Images,imf_i in zip(prompt, feature_,share_ctx_,feature_Image,image):
        # for prompts_i,share_ctxs,imf_i in zip(prompt,share_ctx_,,,image):    
        for prompts_i, feature_i,share_ctxs,feature_Images,imf_i in zip(prompt, feature_,share_ctx_,feature_Image,image):
            text_features, upper_text = self.text_encoder(prompts_i, tokenized_prompts, deep_compound_prompts_text,feature_i)
            upper_text = upper_text / upper_text.norm(dim=-1, keepdim=True)
            upper_texts.append(upper_text)
            
            image_features, clone_feature_image = self.image_encoder(imf_i.unsqueeze(0).type(self.dtype), share_ctxs.type(self.dtype), deep_compound_prompts_vision,feature_Images)
            clone_feature_image = clone_feature_image / clone_feature_image.norm(dim=-1, keepdim=True)
            clone_feature_images.append(clone_feature_image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * image_features @ text_features.t()
            logits.append(l_i)
            image_features_list.append(image_features.squeeze(0))
        logit = torch.cat(logits, dim=0)  
        image_features_list=torch.stack(image_features_list,dim=0)               
########################################################################## 

        
   
   
         
            #image_features = self.image_encoder(image.type(self.dtype), shared_ctx.type(self.dtype), deep_compound_prompts_vision)
            
####################################################################
        
        

        
        
        

        if self.prompt_learner.training:
            # proj_clone = nn.Linear(512, 798)
            # proj_clone.weight = nn.Parameter(compound_prompt_projections[-1].weight.clone().detach(), requires_grad=False)
            # proj_clone.bias = nn.Parameter(compound_prompt_projections[-1].bias.clone().detach(), requires_grad=False)
            cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
            
            distt=[]
            for idnex, item in enumerate(label):
                max_values_cgm=logits_[idnex,item]
                PotenList=find_closest_indexes(logits_[idnex], max_values_cgm, 2)
                #extracted_values = [proj_clone(clone_feature_text[i]) for i in PotenList]
                a_extracted_values=[clone_feature_text[i].to("cuda:1").to(torch.float32) for i in PotenList]
                extracted_values=[]
                for i in a_extracted_values:
                    try:
                        aa=mini_net_(i.to("cuda:1") )
                    except:
                        aa=mini_net_(i.to("cuda:0") )
                    aa = aa / aa.norm( keepdim=True)
                    extracted_values.append(aa.to(clone_feature_images[0]))


                
                stand=torch.mean(cos(clone_feature_images[idnex],extracted_values[0].unsqueeze(0).to(clone_feature_images[idnex]))) 

                
                ek=1-stand.to(torch.float32) 
                       
                        
                     

                distt.append(ek)
            # print(distt)
            distts=torch.stack(distt,dim=0).mean()

            distext=[]
            for idnex, item in enumerate(label):
                max_values_cgm=logits_[idnex,item]
                PotenList=find_closest_indexes(logits_[idnex], max_values_cgm, 2)

                a_extracted_values=[upper_texts[idnex][i].to("cuda:1").to(torch.float32) for i in PotenList]
                extracted_values=[]
                for i in a_extracted_values:
                    try:
                        aa=mini_net_(i.to("cuda:1") )
                    except:
                        aa=mini_net_(i.to("cuda:0") )
                    aa = aa / aa.norm( keepdim=True)
                    extracted_values.append(aa.to(upper_image[0]))


                    
                stand=torch.mean(cos(upper_image[idnex],extracted_values[0].unsqueeze(0).to(upper_image[idnex]))) 

                
                ek=1-stand.to(torch.float32) 

                distext.append(ek)
            # print(distt)
            distexts=torch.stack(distext,dim=0).mean()
                        
                      
            self_informationText=[]


            for text_prompt in [*deep_compound_prompts_text,ctx_ind2d,*compound_prompts_fuxi_text,ctx]:            
                text_prompt = torch.fft.fft2(text_prompt.to(torch.float32),dim=1)
                langToken_entropies=[]
                for langToken in text_prompt:
                    column_entropy = entropyx(langToken)
                    langToken_entropies.append(column_entropy)
                self_informationText.append(langToken_entropies)
                    
            self_informationVision=[]
            for vision_prompt in [*deep_compound_prompts_vision,share_ctx,*deep_compound_prompts_fuxi_vision,image_fuxi_proj_ctx]:
                vision_prompt = torch.fft.fft2(vision_prompt.to(torch.float32),dim=1)
                visionToken_entropies=[]
                for visonToken in vision_prompt:
                        column_entropy = entropyx(visonToken)
                        visionToken_entropies.append(column_entropy)
                self_informationVision.append(visionToken_entropies)
            
            cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
            # kg= cos(text_features_a.to(text_features),text_features)
            # kg  = 1.0-torch.mean(kg)
            # eu=kl_divergence(image_features_list,image_features_.to(text_features))              
            average_difference=calculate_average_differenceTokenwise(self_informationText,self_informationVision)
            ce_loss = F.cross_entropy(logit, label)
            # print("ce:",ce_loss)
            # print("kg:",kg)
            # print("eu:",eu)
            # print("average_difference:",average_difference)
            # print("distts:",distts)

        
            total_loss =  ce_loss           +  average_difference+ 5* distts + 7*distexts
            return total_loss # F.cross_entropy(logits, label)  #total_loss

        return logit
def find_closest_indexes(nums, A, n):
    # 枚举列表，获取索引和值，然后根据与 A 的差值进行排序
    sorted_indexes = sorted(enumerate(nums), key=lambda x: abs(x[1] - A))
    
    # 从排序后的列表中取出前 n 个元素的索引
    closest_indexes = [index for index, _ in sorted_indexes[:n]]
    
    return closest_indexes

def kl_divergence(p_logits, q_logits, epsilon=1e-12):
    """
    计算两个概率分布张量的KL散度损失。
    
    参数:
    p_logits - 第一个概率分布的logits（未归一化的概率预测）。
    q_logits - 第二个概率分布的logits（未归一化的概率预测）。
    epsilon - 用于数值稳定性的小常数。
    
    返回:
    kl_loss - 两个分布之间的KL散度损失。
    """
    # 将logits转换为概率分布
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    
    # 避免log(0)情况，加入小的epsilon
    p = torch.clamp(p, epsilon, 1 - epsilon)
    q = torch.clamp(q, epsilon, 1 - epsilon)
    
    # 计算KL散度
    kl_div = torch.sum(p * torch.log(p / q), dim=-1)
    return kl_div
def calculate_average_differenceTokenwise(A, B):
    """
    Calculate the average difference between corresponding elements of two lists A and B.
    
    Args:
    A (list): The first list with two elements.
    B (list): The second list with two elements.

    Returns:
    float: The average difference.
    """
    differences=[]
    for i , elements in enumerate(A):
        diff=[]
        
        for j in range(len(elements)):
            total_diff= abs(A[i][j] - B[i][j])
            diff.append(total_diff)
        diffMean=torch.mean(torch.stack(diff), dim=0)
        differences.append(diffMean.float())
    
    smooth_differences = torch.mean(torch.stack(differences), dim=0)


    return smooth_differences 

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_clonesX(module, N):
    # 创建一个包含相同模块多次引用的列表
    return [module for _ in range(N)]


@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        #clip_model_ = load_clip_to_cpuO(cfg)


        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            #clip_model_.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model) #,clip_model_

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.to(self.device).mean().backward()
            optim.step()

        loss_summary = {"loss": loss.mean().item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
