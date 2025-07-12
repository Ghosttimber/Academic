import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import sys,os #debug cgm
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #debug cgm
from torch.nn.modules.loss import _Loss


import numpy as np
import gc

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer



_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a type of pet, a photo of a {}.",
    "OxfordFlowers": "a type of flower, a photo of a {}.",
    "FGVCAircraft": "a type of aircraft, a photo of a {}.",
    "DescribableTextures": "a texture of {}.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a type of food, a photo of {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "Cifar10": "a photo of a {}.",
    "Cifar100": "a photo of a {}.",
}


class CLIP(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpuO(cfg)
        clip_model.float()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1,keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1,keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        text_features = self.text_features
        text_features = text_features.to(image_features.device)
        logits = logit_scale * image_features @ text_features.t()
        return logits




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
    for i in im:
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

# def high_pass_filter(size: int, cutoff: int):
#     """
#     Create a 2D high-pass filter using a Gaussian window.
#     :param size: The size of the filter (should be odd).
#     :param cutoff: Cutoff frequency for the filter.
#     :return: A 2D high-pass filter.
#     """
#     rows = torch.fft.fftfreq(size)
#     cols = torch.fft.fftfreq(size)
#     radius = torch.sqrt((rows[:, None])**2 + (cols[None, :])**2)
#     filter = torch.exp(-radius**2 / (2 * (cutoff/2.0)**2))
#     return torch.tensor(1.0 - filter, dtype=torch.float32)


# def low_pass_filter(size, cutoff, taper_width=0.1):
#     rows = torch.fft.fftfreq(size[0])
#     cols = torch.fft.fftfreq(size[1])
#     radius = torch.sqrt(rows[:, None]**2 + cols[None, :]**2)
#     filter = torch.exp(-(radius - cutoff)**2 / (2 * taper_width**2))
#     return filter

# def low_pass_process(image, cutoff) :

#     lists=[]
#     for image_tensor in image:

#         f_image = torch.fft.fftn(image_tensor)
#         filter = low_pass_filter(image_tensor.shape[-2:], cutoff)
#         filter = filter.type_as(image_tensor)
#         f_image_lp = f_image * filter
#         image_lp = torch.fft.ifftn(f_image_lp).real


#         lists.append(image_lp)
    
#     stacked_img=torch.stack(lists,dim=0)
    
#     return stacked_img  


# def extract_high_frequency(image: torch.Tensor, cutoff: int) -> torch.Tensor:
#     """
#     Extract high frequency features from an image using Fourier transform.
#     :param image: A 2D image tensor.
#     :param cutoff: Cutoff frequency for the filter.
#     :return: Image with high frequency features.
#     """
#     # Apply Fourier transform
#     f_transform = torch.fft.fftn(image)
    
#     # Move zero frequency component to the center
#     f_transform_shifted = torch.fft.fftshift(f_transform)
    
#     # Apply high-pass filter
#     filter = high_pass_filter(image.size(-1), cutoff).cuda()
#     f_transform_shifted_filtered = f_transform_shifted.cuda() * filter
    
#     # Inverse shift
#     f_transform_filtered = torch.fft.ifftshift(f_transform_shifted_filtered)
    
#     # Inverse Fourier transform
#     image_high_freq = torch.fft.ifftn(f_transform_filtered)
    
#     # Return the real part of the inverse Fourier transform
#     return image_high_freq.real

# def ssim(img1, img2, window,window_size=11, k1=0.01, k2=0.03, C1=None, C2=None):
#     """
#     Compute SSIM between two images.
#     """
#     if not C1:
#         C1 = (k1 * 255) ** 2
#     if not C2:
#         C2 = (k2 * 255) ** 2
        
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=3)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=3)
    
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=3) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=3) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=3) - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     return ssim_map.mean()


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
    design_details = {"trainer": 'APLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "aple_length": cfg.TRAINER.APLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def load_clip_to_cpuO(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_modelO(state_dict or model.state_dict())

    return model




# def weak_augmentation(image):
#     # 随机水平翻转（左右翻转）
#     # if np.random.rand() < 0.5:
#     #     image = cv2.flip(image, 1)

#     # 随机旋转
#     angle = np.random.randint(-15, 15)
#     rows, cols, _ = image.shape
#     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
#     image = cv2.warpAffine(image, M, (cols, rows))

#     # 随机平移
#     tx = np.random.randint(-20, 20)
#     ty = np.random.randint(-20, 20)
#     M = np.float32([[1, 0, tx], [0, 1, ty]])
#     image = cv2.warpAffine(image, M, (cols, rows))

#     # 随机缩放
#     scale_factor = np.random.uniform(0.8, 1.2)
#     image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

#     return image

# def strong_augmentation(image):
#     device=image.device
#     # 随机裁剪
#     crop_size = np.random.randint(50, 100)
#     x = np.random.randint(0, image.shape[1] - crop_size)
#     y = np.random.randint(0, image.shape[2] - crop_size)
#     cropped_image = image[:, y:y+crop_size, x:x+crop_size]

#     # 随机调整亮度和对比度
#     alpha = 1.0 + np.random.uniform(-0.2, 0.2)
#     beta = np.random.uniform(-20, 20)
#     enhanced_image = alpha * cropped_image + beta

#     # 添加高斯噪声
#     noise = np.random.normal(0, 10, enhanced_image.cpu().shape).astype(np.float32)
#     noisy_image = enhanced_image.cpu() + noise

#     return noisy_image.to(device)

# class DnCNN(nn.Module):
#     def __init__(self, channels=768, num_layers=4):
#         super(DnCNN, self).__init__()
#         layers = []
        
#         # First layer
#         layers.append(nn.Conv2d(3, channels, kernel_size=3, padding=1))
#         layers.append(nn.ReLU(inplace=True))
        
#         # Middle layers
#         for _ in range(num_layers - 2):
#             layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
#             layers.append(nn.BatchNorm2d(channels))
#             layers.append(nn.ReLU(inplace=True))
        
#         # Last layer
#         layers.append(nn.Conv2d(channels, 1, kernel_size=3, padding=1))
        
#         self.dncnn = nn.Sequential(*layers)
    
#     def forward(self, x):
#         noise = self.dncnn(x)
#         return x - noise  # Subtracting the predicted noise from the input


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
   
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
           
class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.APLE.N_CTX
        ctx_init = cfg.TRAINER.APLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.APLE.PROMPT_DEPTH >= 1, "For APLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.APLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('APLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of APLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        # first 0
        # with torch.no_grad():
        self.compound_prompts_text_ctx = nn.Parameter(ctx_vectors.cuda())
        ctxI_vectors = torch.empty(n_ctx, 768, dtype=dtype) 
        nn.init.normal_(ctxI_vectors, std=0.02)
        
        self.compound_prompts_vision_shareI = nn.Parameter(ctxI_vectors.cuda())

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

 
        # compound prompts
###################################   experiments ###################################################
######### previous
        # self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
        #                                               for _ in range(self.compound_prompts_depth - 1)])
        # for single_para in self.compound_prompts_text:
        #     nn.init.normal_(single_para, std=0.02)
        # # Also make corresponding projection layers, for each prompt

######### present

        # Also make corresponding projection layers, for each prompt
        
        ####ctxI_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) final =4 *####

        
        # self.compound_prompts_vision_shareI = nn.Parameter(ctxI_vectors).to(self.ctx) ## final =4 *####
        # alpha=0.5
        # self.share=alpha*self.ctxI+(1-alpha)*self.ctx 
        #self.share=alpha*self.ctxI+(1-alpha)*self.ctx  # self.share=alpha*self.ctxI+(1-alpha)*self.ctx 

        #self.shareI=self.proj(self.share.float()).half().to(self.ctx)
        #self.shareI=self.proj(self.ctxI.float()).half().to(self.ctx)

        
        
        # self.compound_prompts_text = nn.ParameterList([self.ctx ## !!! self.share
        #                                               for _ in range(self.compound_prompts_depth - 1)])   
        
        self.compound_prompts_vision = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.empty(n_ctx, 768), std=0.02)).cuda()
                                                      for _ in range(self.compound_prompts_depth - 1)])

        self.compound_prompts_text = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.empty(n_ctx, 512), std=0.02)).to(self.compound_prompts_vision[0])
                                                      for _ in range(self.compound_prompts_depth - 1)])   
        
        # listI=[]
        # for index,i  in enumerate(self.compound_prompts_text):
        #     listI.append(self.ctxI.to(self.shareI)*alpha+(1-alpha)*self.compound_prompts_text[index].to(self.shareI))
        
        # self.compound_prompts_vision = nn.ParameterList(listI)  
        
           
        # single_layer = nn.Linear(ctx_dim, 768).float()

###################################   experiments ###################################################
        
        
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

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
###################################   experiments ###################################################
######### previous
        #ctx = self.ctx
######### present       
        prefix = self.token_prefix
        suffix = self.token_suffix 
        ctx = self.compound_prompts_text_ctx.to(suffix) #share.to(suffix)
        compound_prompts_vision_shareI=self.compound_prompts_vision_shareI.to(ctx)
        compound_prompts_text=self.compound_prompts_text

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)


        prompts = self.construct_prompts(ctx, prefix, suffix) # first 2

        # Before returning, need to transform
        # prompts to 768 for the visual side
###################################   experiments ###################################################
        
        visual_deep_prompts = []
        for index in range(self.compound_prompts_depth - 1):
            #### visual_deep_prompts.append(layer(self.compound_prompts_vision[index].float()).half().to(prompts) )
            visual_deep_prompts.append(self.compound_prompts_vision[index].float().half().to(prompts) )
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, compound_prompts_vision_shareI, compound_prompts_text, visual_deep_prompts  # pass here original, as for visual 768 is required
    
    
###################################   experiments ###################################################
# class BetaNetwork(nn.Module):
#     def __init__(self):
#         super(BetaNetwork, self).__init__()
#         self.beta = nn.Parameter(torch.tensor([0.5]))  # Initialize beta as 0.5

#     def forward(self, image, imagefft):
#         return self.beta * image + (1 - self.beta) * imagefft

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_class=len(classnames)

        

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        
          
        
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)

        imageprocess= True
        if imageprocess:
            
            
            
            #####  image process
            #cutoff_frequency = 29  # 15
            #imagefft = extract_high_frequency(image, cutoff_frequency) 
            
            sigma=0.05 # 0.1 0.4 0.1 
            ######0.05-euros
            imagefft = filter_frequency(image, sigma) 
            
            # cutoff_frequency=0.15
            # imagefft = low_pass_process(image, cutoff_frequency) 
            #####  fusion
            
            #imagefft=        self.DnCNN(image)
            # imageC=self.betanet(image,imagefft)
            beta=0.9
            imageC=beta*image+(1-beta)*imagefft
            # imageC=image
            #####   MSE
            # image_featuresO = self.image_encoder(image.type(self.dtype), shared_ctx, [])  
            # image_features1 = self.image_encoder(imagefft.type(self.dtype), shared_ctx, [])
            # mse = ((image_featuresO - image_features1)** 2).mean(dim=(1,2)).sum()
            

            
            image_features = self.image_encoder(imageC.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        else:
            
            image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()


    
        if self.prompt_learner.training:
            return logits 
        

        return logits
    
# CUSTOM_TEMPLATES = {
#     "OxfordPets": "a type of pet, a photo of a {}.",
#     "OxfordFlowers": "a type of flower, a photo of a {}.",
#     "FGVCAircraft": "a type of aircraft, a photo of a {}.",
#     "DescribableTextures": "a texture of {}.",
#     "EuroSAT": "a centered satellite photo of {}.",
#     "StanfordCars": "a photo of a {}.",
#     "Food101": "a type of food, a photo of {}.",
#     "SUN397": "a photo of a {}.",
#     "Caltech101": "a photo of a {}.",
#     "UCF101": "a photo of a person doing {}.",
#     "ImageNet": "a photo of a {}.",
#     "ImageNetSketch": "a photo of a {}.",
#     "ImageNetV2": "a photo of a {}.",
#     "ImageNetA": "a photo of a {}.",
#     "ImageNetR": "a photo of a {}.",
#     "Cifar10": "a photo of a {}.",
#     "Cifar100": "a photo of a {}.",
# }   
    

    
class ProGradLoss(_Loss):
    def __init__(self, T):
        super(ProGradLoss, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits, label):
        xe_loss = F.cross_entropy(stu_logits, label)

        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T,
                                            -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return xe_loss, kl_loss
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# class MulticlassFocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=None,classnames=1):
#         super(MulticlassFocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha  # 可选的alpha参数，用于调整类别权重
#         self.num_classes=classnames
#     def forward(self, inputs, targets):
#         num_classes = self.num_classes
        
#         log_probs = F.log_softmax(inputs, dim=1)
#         probs = torch.exp(log_probs)
        
#         # 构建 one-hot 编码的标签
#         targets_one_hot = F.one_hot(targets, num_classes)
        
#         # 计算交叉熵损失
#         cross_entropy = -targets_one_hot * log_probs

#         focal_loss = torch.pow(1 - probs, self.gamma) * cross_entropy
        
#         return focal_loss.mean()

def calculate_angle(tensor1, tensor2):
    # 计算两个张量的点积
    dot_product = torch.dot(tensor1.flatten(), tensor2.flatten())

    # 计算每个张量的范数
    norm_tensor1 = torch.norm(tensor1)
    norm_tensor2 = torch.norm(tensor2)

    # 计算两个张量之间的夹角（以弧度为单位）
    cos_angle = dot_product / (norm_tensor1 * norm_tensor2)
    
    # 将弧度转换为角度
    angle = torch.acos(cos_angle) * (180 / torch.pi)

    return angle  

# class ExponentialMovingAverage:
#     def __init__(self, model, decay):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}

#         for name, param in self.model.named_parameters():
#             if param.requires_grad and  'compound_prompts_text'   in name:
#                 self.shadow[name] = param.data.clone()

#     def update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and  'compound_prompts_text'  in name:
#                 if name in self.shadow:
#                     self.shadow[name] -= (1.0 - self.decay) * (self.shadow[name] - param.data)
#                 else:
#                     self.shadow[name] = param.data.clone()

#     def apply(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and  'compound_prompts_text'  in name:
#                 param.data = self.shadow[name]
                 
@TRAINER_REGISTRY.register()
class APLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.APLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):

        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.APLE.PREC == "fp32" or cfg.TRAINER.APLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model_nograd = CLIP(cfg, classnames)
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model_nograd.named_parameters():
            param.requires_grad_(False)
            
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
        

        self.model_nograd.to(self.device)
        for name, param in self.model_nograd.named_parameters():
            param.requires_grad_(False)
        # self.model_nograd = self.model_nograd.to('cuda:1')
        parameters_to_modify = []
        self.criterion = ProGradLoss(T=1)

        # 
        for name, param in self.model.named_parameters():
            if name != "compound_prompts_vision" and  param.requires_grad:  # 排除名称为 "sb" 的参数
                parameters_to_modify.append(param)
                
        fisrt_parm=[param for name, param in self.model.named_parameters() if 'compound_prompts_vision' in name]
        self.optim = build_optimizer(fisrt_parm, cfg.OPTIM) #### cgm      
        # self.optim_vision = build_optimizer(self.model, cfg.OPTIM,param_groups=self.model.parameters('compound_prompts_vision')) compound_prompts_text
        self.params_to_update = [param for name, param in self.model.named_parameters() if 'compound_prompts_text'  in name ]
        self.optim_2n = build_optimizer(self.params_to_update , cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # self.sched_vision = build_lr_scheduler(self.optim_vision, cfg.OPTIM)

        
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.APLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.model_nograd = nn.DataParallel(self.model_nograd)


    def forward_backward(self, batch,epoch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim_2n=self.optim_2n
        optim = self.optim


        scaler = self.scaler


        

        prec = self.cfg.TRAINER.APLE.PREC
        if prec == "amp":
            with autocast():
                loss,FocalLoss = model(image, label)
            optim.zero_grad()
            scaler.scale(CEloss).backward()
            scaler.step(optim)
            scaler.update()
            

            
            
        else:
            lambda_text=0.5
            lambda_=0.4
                        
              
            output = model(image, label)      

            #params="compound_prompts_text"  ###compound_prompts_vision

            with torch.no_grad():
                # zs_clip_output = self.model_nograd(image,label)
                zs_clip_output = self.model_nograd(image)
                
            xe_loss, kl_loss = self.criterion(output,zs_clip_output.detach(),label)                

            optim_2n.zero_grad() 
            loss=xe_loss+lambda_text*kl_loss
            loss.mean().backward(retain_graph=True)
            
            # kl_loss=kl_loss.mean()           
            # kl_loss.backward(retain_graph=True)
            # b_grads = [param.grad.clone() for name, param in model.named_parameters() if  param.grad is not None]
            # optim_2n.zero_grad()   
            # xe_loss=xe_loss.mean()
            # xe_loss.backward(retain_graph=True)
            # for p, b_grad in zip( [ param for name, param in model.named_parameters() if   param.grad is not None], b_grads):
            #     # calculate cosine distance
            #         a_grad = p.grad.clone()
            #         p.grad = a_grad + lambda_text *b_grad 



            optim_2n.step()
            optim.zero_grad() 
            optim_2n.zero_grad() 

            del kl_loss
            del xe_loss
            del loss
            del output
            
            torch.cuda.empty_cache()  # 如果在 GPU 上运行
            gc.collect()




            output = model(image, label)                      
            xe_loss, kl_loss = self.criterion(output,zs_clip_output.detach(),label)                
            # params="vision"  ###compound_prompts_vision  
            # optim_2n.zero_grad() 
            # loss =xe_loss+lambda_text*kl_loss
            # loss=loss.mean()
            # loss.backward(retain_graph=True)
            optim.zero_grad()
            
            loss=xe_loss+lambda_*kl_loss
            loss.mean().backward(retain_graph=True)
            # kl_loss=kl_loss.mean()
            # kl_loss.backward(retain_graph=True)


            # c_grads = [param.grad.clone() for name, param in model.named_parameters() if  param.grad is not None]

            # optim.zero_grad()          
            # xe_loss=xe_loss.mean()
            # xe_loss.backward(retain_graph=True)
            # for v, c_grad in zip( [ param for name, param in model.named_parameters() if param.grad is not None], c_grads):
            #     # calculate cosine distance

            #         v_grad = v.grad.clone()
            #         v.grad = v_grad + lambda_*c_grad
                    
            optim.step()
            optim.zero_grad() 
            optim_2n.zero_grad() 
            del kl_loss
            del loss
            del xe_loss
            del output
            torch.cuda.empty_cache()  # 如果在 GPU 上运行
            gc.collect()
    
            
            output = model(image, label)
            loss=F.cross_entropy(output, label)
            optim.zero_grad() 
            optim_2n.zero_grad() 
            loss=loss.mean()
            loss.backward(retain_graph=True)
            
            optim.step()
            optim_2n.step()


            del output
            torch.cuda.empty_cache()  # 如果在 GPU 上运行
            gc.collect()
            optim.zero_grad() 
            optim_2n.zero_grad() 
######################################################################################################################                 
        loss_summary = {
            "loss": loss.item()
        }

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
