import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchmodels
import numpy as np
#from loss import dice_loss
from models.losses import dice_loss
#from IPython.core.debugger import set_trace

class InstanceUNet(nn.Module):
    def __init__(self, n_class, emb_features=128):
        super().__init__()

        self.base_model = ResNetUNet(n_class=None)
        in_features = 128
        self.class_segmentation = nn.Conv2d(in_features, n_class, 1)
        self.instance_embedding = nn.Sequential(
            nn.Conv2d(in_features, emb_features, 1),
            nn.Sigmoid()
        )

        self.margin = 1.0

    def forward(self, input, gt_class_masks=None, gt_instance_masks=None):
        outputs = {}
        losses = {}

        shared_feature_maps = self.base_model(input)
        outputs['shared_feature_maps'] = shared_feature_maps


#         set_trace()
        class_masks = self.class_segmentation(shared_feature_maps)
        outputs['class_masks'] = class_masks

        if gt_class_masks is not None:
            bce = F.binary_cross_entropy_with_logits(class_masks, gt_class_masks)
            losses['bce'] = bce

            class_masks = F.sigmoid(class_masks)
            dice = dice_loss(class_masks, gt_class_masks)
            losses['dice'] = dice
            loss = bce + dice

        instance_embedding = self.instance_embedding(shared_feature_maps)
        outputs['instance_embedding'] = instance_embedding
        
        if gt_instance_masks is not None:
            within_loss = []
            between_loss = []
            between_dist = []
            for b in range(input.size()[0]):
                cluster_centroids, mean_within_distances = cluster_distances(instance_embedding[b], gt_instance_masks[b])
                within_loss.append(mean_within_distances.mean())

                dist_matrix = pairwise_dist(cluster_centroids)
                upper_mask = torch.triu(torch.ones_like(dist_matrix).type(torch.ByteTensor), diagonal=1)

                # hinge loss
                between_loss.append(F.relu(self.margin - dist_matrix[upper_mask]).mean())
                between_dist.append(dist_matrix[upper_mask].mean())

            within_loss = torch.stack(within_loss).mean()
            between_loss = torch.stack(between_loss).mean()
            between_dist = torch.stack(between_dist).mean()

            losses['within'] = within_loss
            if np.isnan(between_loss.detach().cpu().numpy()):
                set_trace()
            else:
                losses['between'] = between_loss
            losses['between_dist'] = between_dist

            loss = losses['bce'] + losses['dice'] + losses['within'] + losses['between']

        if gt_class_masks is not None:
            losses['loss'] = loss

        return outputs, losses


def pairwise_dist(vectors):
    l2norm = vectors.pow(2).sum(dim=1)
    distance_matrix = l2norm.view(1, -1) + l2norm.view(-1, 1) - 2 * torch.mm(vectors, vectors.t())

    return torch.clamp(distance_matrix, 0.0, np.inf)

def cluster_distances(embedding_map, gt_instance_masks):
    # embedding_map: (ch, h, w)
    # gt_instance_masks: (# instances, h, w)
    cluster_centroids = []
    mean_within_distances = []
    
#     embedding_map[embedding_map >= 0.5] = 1
#     embedding_map[embedding_map < 0.5] = 0

    for k in range(gt_instance_masks.shape[0]):
        instance_mask = gt_instance_masks[k] > 0.5

        vectors_within_cluster = embedding_map[:, instance_mask].t()
        vector_count = vectors_within_cluster.shape[0]

        cluster_centroid = vectors_within_cluster.mean(0)
        cluster_centroids.append(cluster_centroid)

        mean_within_distance = (vectors_within_cluster - cluster_centroid).pow(2).sum(dim=1).mean()
        mean_within_distances.append(mean_within_distance)

    mean_within_distances = torch.stack(mean_within_distances, 0)
    cluster_centroids = torch.stack(cluster_centroids, 0)

    return cluster_centroids, mean_within_distances



def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )











class ObjectDetectUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = ResNetUNet(n_class=None)
#         in_features = 128
        in_features = 256        
        self.class_segmentation = nn.Conv2d(in_features, 2*n_class, 1)
        self.n_class = n_class
        

    def forward(self, input, gt_class_masks=None, gt_centers=None):
        outputs = {}
        losses = {}

        shared_feature_maps = self.base_model(input)
        outputs['shared_feature_maps'] = shared_feature_maps


#         set_trace()
        all_masks = self.class_segmentation(shared_feature_maps)
        class_masks = all_masks[:,:self.n_class,:,:]
        center_masks = all_masks[:,self.n_class:,:,:]
        
        #         print(class_masks.shape)
        outputs['class_masks'] = class_masks
        outputs['center_masks'] = center_masks

        if gt_class_masks is not None:
            bce_mask = F.binary_cross_entropy_with_logits(class_masks, gt_class_masks)
            losses['bce_mask'] = bce_mask

            bce_center = F.binary_cross_entropy_with_logits(center_masks, gt_centers)
            losses['bce_center'] = bce_center


            class_masks = F.sigmoid(class_masks)
            dice = dice_loss(class_masks, gt_class_masks)
            losses['dice'] = dice
            loss = bce_mask + bce_center + dice
#             loss = bce_center 
            
            losses['loss'] = loss

            
            
            
        return outputs, losses
            
            
        


    
    

    
class WidthHeightUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = ResNetUNet(n_class=None)
#         in_features = 128
        in_features = 256        
        self.class_segmentation = nn.Conv2d(in_features, 4*n_class, 1)
        self.n_class = n_class
        

    def forward(self, input, gt_class_masks=None, gt_centers=None, gt_widths=None, gt_heights=None):
        outputs = {}
        losses = {}
        
        
        shared_feature_maps = self.base_model(input)
        outputs['shared_feature_maps'] = shared_feature_maps

        all_masks = self.class_segmentation(shared_feature_maps)
        class_masks = all_masks[:,:self.n_class,:,:]
        center_masks = all_masks[:,self.n_class:2*self.n_class,:,:]
        width_masks = all_masks[:,2*self.n_class:3*self.n_class,:,:]
        height_masks = all_masks[:,3*self.n_class:,:,:]

        outputs['class_masks'] = class_masks
        outputs['center_masks'] = center_masks
        outputs['width_masks'] = width_masks
        outputs['height_masks'] = height_masks

        
        if gt_class_masks is not None:
            
            gt_class_masks = gt_class_masks.type(torch.cuda.FloatTensor)  
            gt_centers = gt_centers.type(torch.cuda.FloatTensor)  
            gt_widths = gt_widths.type(torch.cuda.FloatTensor)  
            gt_heights = gt_heights.type(torch.cuda.FloatTensor)  
            
            
            
            
            bce_mask = F.binary_cross_entropy_with_logits(class_masks, gt_class_masks)
            losses['bce_mask'] = bce_mask

            class_masks = F.sigmoid(class_masks)
            dice = dice_loss(class_masks, gt_class_masks)
            losses['dice'] = dice
            


            bce_center = F.binary_cross_entropy_with_logits(center_masks, gt_centers)
            losses['bce_center'] = bce_center
            
            center_masks = F.sigmoid(center_masks)
            dice_center = dice_loss(center_masks, gt_centers)
            losses['dice_center'] = dice_center
            

            

            width_masks = (gt_centers > 0).type(torch.cuda.FloatTensor) * width_masks
            criterion = torch.nn.MSELoss(reduction='elementwise_mean') #'mean'
            l2_width = criterion(width_masks, gt_widths)
            losses['l2_width'] = l2_width

            height_masks = (gt_centers > 0).type(torch.cuda.FloatTensor) * height_masks            
            l2_height = criterion(height_masks, gt_heights)
            losses['l2_height'] = l2_height
            
            
            loss = bce_mask + dice +bce_center +  l2_width + l2_height + dice_center
#             loss = bce_center 
            
            losses['loss'] = loss

            
            
            
        return outputs, losses
            

















class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchmodels.resnet18(pretrained=True)

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
#         self.conv_original_size2 = convrelu(64 + 128, 128, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 256, 3, 1)

#         self.final = None if n_class is None else nn.Conv2d(128, n_class, 1)
        self.final = None if n_class is None else nn.Conv2d(256, n_class, 1)


    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        if self.final:
            x = self.conv_last(x)

        return x
