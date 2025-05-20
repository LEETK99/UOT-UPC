import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    cuda_index = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index

import torch
import torch.nn as nn



sys.path.append('../util')
sys.path.append('..')
from util import *
from base_model_util import *
import pointnet2_model_api as PN2 
from pointnet2_ops.pointnet2_utils import QueryAndGroup


# Generator
class USSPA_G_T1(nn.Module):
    def __init__(self):
        super().__init__()
        self.GPN = 512
        self.E_R = PcnEncoder2(out_c=512)
        self.D_R = MlpConv(512, [512, 512, 1024, 1024, self.GPN*3])

        self.mlp_mirror_ab = MlpConv(512, [128, 128, 2])
    
    def get_mirror(self, point, ab):
        __e = 1e-8
        A, B = torch.split(ab, [1, 1], 1)

        x = point[:,:,0:1]
        z = point[:,:,2:3]

        AxBz = 2*(A*x+B*z)/(A**2+B**2+__e)

        new_x = x - A*AxBz
        new_z = z - B*AxBz

        y = point[:,:,1:2]
        point = torch.cat([new_x, y, new_z], 2)
        return point

    def forward(self, partial_input, complete_input):
        B, _, _ = partial_input.shape
        input = torch.cat([partial_input, complete_input], 0)

        f_R_0 = self.E_R(input)  
        point_R_0 = self.D_R(f_R_0)
        point_R_0 = point_R_0.reshape([-1, self.GPN, 3])

        ab = torch.tanh(self.mlp_mirror_ab(f_R_0))
        input_R_M = self.get_mirror(input, ab)  
        
        output = torch.cat([input, input_R_M, point_R_0], 1) 
        output = PN2.FPS(output, 2048)
        partial_output, complete_output = torch.split(output, [B, B], 0)  


        return partial_output, complete_output
    
class USSPA_G_T2(nn.Module):
    def __init__(self):
        super().__init__()
        self.GPN = 512
        self.E_A = PcnEncoder2(out_c=512)
        self.D_A = MlpConv(512, [512, 512, 1024, 1024, self.GPN*3])

        self.mlp_mirror_ab = MlpConv(512, [128, 128, 2])

        self.mlp_refine_1 = MlpConv(3, [256, 256, 256])
        self.mlp_refine_2 = MlpConv(512, [256, 256, 256])
        self.qg = QueryAndGroup(0.25, 32)

        self.UPN = 4
        
        self.mlp_refine_3 = MlpConv(512, [256, 256, 256])

        self.mlp_refine_4 = MlpConv(256+3, [512, 512, 3*self.UPN]) 
        
        self.tanh = torch.nn.Tanh()

    def upsampling_refine(self, point):
        #### encode feature ####
        B, N, _ = point.shape
        
        x = self.mlp_refine_1(point.permute(0, 2, 1))
        x_max = torch.max(x, 2, keepdim=True).values
        x = self.mlp_refine_2(torch.cat([x, x_max.repeat([1, 1, N])], 1))
        x_local = self.qg(point, point, x)[:,3:,:,:]      
        x_local = torch.max(x_local, -1).values         
        x = self.mlp_refine_3(torch.cat([x, x_local], 1))

        #### upsampling refine ####      
        x = torch.cat([point.permute(0, 2, 1), x], 1)
       
        shift = self.mlp_refine_4(x)
      

        #### shift ####
        res = torch.unsqueeze(point, 2).repeat([1, 1, self.UPN, 1])
        res = torch.reshape(res, [B, -1, 3])
        shift = shift.permute(0, 2, 1).reshape([B, -1, 3])
        res = res + shift

        return res
    
    def forward(self, input_partial, input_complete):
        B, N, _ = input_partial.shape

        x = torch.cat([input_partial, input_complete], 0)
        x = self.E_A(x)
        x = self.D_A(x)
        x = x.reshape([-1, self.GPN, 3])
       
        x = self.upsampling_refine(x)
        
        output_partial, output_complete = torch.split(x, [B, B], 0)

        return output_partial, output_complete


# Discrminator
class PointDIS(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = PcnEncoder2(out_c=1024) 
    
        self.mlp = MlpConv(1024, [256, 256, 128, 128, 1]) 
        
    def forward(self, point):
        d_p = self.encoder(point)
        d_p = self.mlp(d_p)
        d_p = d_p[:,0,0]
        return d_p

class USSPA_D_split(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_f = MlpConv(512, [64, 64, 1])
        self.d_p = PointDIS()

    def forward(self, T1_partial_output, T2_partial_output, T1_complete_output, T2_complete_output, input_complete):
        B = T1_partial_output.shape[0]
        
        input_complete.requires_grad = True
        point = torch.cat([T1_partial_output, T2_partial_output, T1_complete_output, T2_complete_output, input_complete], 0)
        d_p = self.d_p(point)

        V1_partial, V2_partial, V1_complete, V2_complete, V_real = torch.split(d_p, [B, B, B, B, B], 0)

        return V1_partial, V2_partial, V1_complete, V2_complete, V_real
    
# Discrminator
class PointDIS_OT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PcnEncoder2(out_c=512)
       
        self.mlp = MlpConv(512, [128, 128, 1])
        
    def forward(self, point):
        d_p = self.encoder(point)
        d_p = self.mlp(d_p)
        d_p = d_p[:,0,0]
        return d_p

class USSPA_D_split_OT(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_f = MlpConv(512, [64, 64, 1])
        self.d_p = PointDIS_OT()

    def forward(self, T1_partial_output, T2_partial_output, T1_complete_output, T2_complete_output, input_complete):
        B = T1_partial_output.shape[0]
        
        input_complete.requires_grad = True
        point = torch.cat([T1_partial_output, T2_partial_output, T1_complete_output, T2_complete_output, input_complete], 0) 
        d_p = self.d_p(point)

        V1_partial, V2_partial, V1_complete, V2_complete, V_real = torch.split(d_p, [B, B, B, B, B], 0)

        return V1_partial, V2_partial, V1_complete, V2_complete, V_real
    
    
class USSPA_D_split_solo(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_f = MlpConv(512, [64, 64, 1])
        self.d_p = PointDIS()
    
    def forward(self, input_R_point_R_0, input_A, input_A_point_A_0): 
        B = input_R_point_R_0.shape[0]
        point = torch.cat([input_R_point_R_0, input_A, input_A_point_A_0], 0) 
        d_p = self.d_p(point)
        d_p_R, d_p_A, d_p_A_0 = torch.split(d_p, [B, B, B], 0)

        return d_p_R, d_p_A, d_p_A_0


class USSPA_split(nn.Module):
    def __init__(self, args, dis = 0.03):
        super().__init__()
        if args.TC=='T11':
            self.G1 = USSPA_G_T1()
            self.G2 = USSPA_G_T1()
            self.D = USSPA_D_split()    
        elif args.TC=='T12':
            self.G1 = USSPA_G_T1()
            self.G2 = USSPA_G_T2()
            self.D = USSPA_D_split()    
        elif args.TC=='T22':
            self.G1 = USSPA_G_T2()
            self.G2 = USSPA_G_T2()
            self.D = USSPA_D_split()    
        elif args.TC=='T1':
            self.G1 = USSPA_G_T1()
            self.D = USSPA_D_split_solo() 
        elif args.TC=='T2':
            self.G1 = USSPA_G_T2()
            self.D = USSPA_D_split_solo()
        elif args.TC=='T12_OT':
            self.G1 = USSPA_G_T1()
            self.G2 = USSPA_G_T2()
            self.D = USSPA_D_split_OT()
        else:
            raise NotImplementedError 
        
        self.typeG = args.TC
        self.cut_grad = args.cut_grad
        self.motivation = args.motivation
    
    def forward(self, data):
        partial_data, complete_data = data

        input_partial = partial_data[0]
        input_complete = complete_data[0]

        T1_partial_output, T1_complete_output = self.G1(input_partial, input_complete)
        if self.typeG in ['T1', 'T2']:
            V1_partial, V1_complete, V_real = self.D(T1_partial_output, T1_complete_output, input_complete)
            T2_partial_output, T2_complete_output, V2_partial, V2_complete = None, None, torch.zeros_like(V1_partial), torch.zeros_like(V1_complete)
        else:
            if self.cut_grad:
                T2_partial_output, T2_complete_output = self.G2(T1_partial_output.detach(), T1_complete_output.detach())
            else:
                T2_partial_output, T2_complete_output = self.G2(T1_partial_output, T1_complete_output)
            V1_partial, V2_partial, V1_complete, V2_complete, V_real = self.D(T1_partial_output, T2_partial_output, T1_complete_output, T2_complete_output, input_complete)
            
        samples = [input_partial, input_complete, T1_partial_output, T1_complete_output, T2_partial_output, T2_complete_output]
        values = [V1_partial, V1_complete, V2_partial, V2_complete, V_real]

        return samples, values