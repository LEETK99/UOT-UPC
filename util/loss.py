from loss_util import *
import torch.nn.functional as F
import ot

def select_phi(name, alpha=1):
    if name == 'linear':
        def phi(x):
            return x
            
    elif name == 'kl':
        def phi(x):
            return alpha * torch.exp(x/alpha)
            
    elif name == 'softplus':
        torch.sigmoid
        def phi(x):
            return 2 * alpha * (F.softplus(x / alpha) - F.softplus(0*x))
    else:
        raise NotImplementedError
    
    return phi
distance = ChamferDistance()

def select_cost_function(name):
    if name == 'cd1':
        def cost_function(p1, p2):
            p2g, g2p = distance(p1, p2)
            p2g = torch.sqrt(p2g)
            g2p = torch.sqrt(g2p)
            p2g = torch.mean(p2g, 1)
            g2p = torch.mean(g2p, 1)
            cd = (p2g + g2p)/2
            return cd
        
    elif name =='cd':
        def cost_function(p1, p2):
            p2g, g2p = distance(p1, p2)
            p2g = torch.mean(p2g, 1)
            g2p = torch.mean(g2p, 1)
            cd = p2g + g2p
            #return cd, p2g, g2p
            return cd
        
    elif name == 'infocd':
        def cost_function(p1, p2):
            dist1, dist2 = distance(p1, p2)
            dist1 = torch.clamp(dist1, min=1e-9)
            dist2 = torch.clamp(dist2, min=1e-9)
            d1 = torch.sqrt(dist1)
            d2 = torch.sqrt(dist2)
        
            distances1 = - torch.log(torch.exp(-0.5* d1)/(torch.sum(torch.exp(-0.5* d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
            distances2 = - torch.log(torch.exp(-0.5 * d2)/(torch.sum(torch.exp(-0.5* d2) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
        
            return (torch.sum(distances1, 1) + torch.sum(distances2, 1)) / 2
        
    elif name == 'l2':
        def cost_function(p1, p2):
            diff = p1-p2
            diff_2 = (torch.sum(diff*diff, dim=2))
            diff_2 = torch.mean(diff_2, dim=1)

            return diff_2
        
    elif name == 'emd' :
        def cost_function(p1, p2, reg=0.1):
            uniform = torch.ones(len(p1[0]), device=p1.device) / len(p1[0])
            M = ((p1[0][None,:] - p2[0][:,None])**2).sum(2)
            return [ot.sinkhorn2(uniform, uniform, M, reg=reg, numItermax=10)]
        
    elif name == 'cd_fwd':
        def cost_function(p1, p2):
            p2g, g2p = distance(p1, p2)
            p2g = torch.mean(p2g, 1)
            g2p = torch.mean(g2p, 1)
            cd = p2g + g2p
            #print(cd)
            return p2g
    else:
        raise NotImplementedError
    return cost_function


def cd1(p1, p2):
    p2g, g2p = distance(p1, p2)
    p2g = torch.sqrt(p2g)
    g2p = torch.sqrt(g2p)
    p2g = torch.mean(p2g, 1)
    g2p = torch.mean(g2p, 1)
    cd = (p2g + g2p)/2
    return cd
def density_loss(x):
    x1 = x.unsqueeze(1)
    x2 = x.unsqueeze(2)
    diff = (x1-x2).norm(dim=-1)
    diff, idx = diff.topk(16, largest=False)
    # print(idx.shape)
    loss = diff[:,:,1:].mean(2).std(1)
    return loss
        
def cd(p1, p2):
    p2g, g2p = distance(p1, p2)
    p2g = torch.mean(p2g, 1)
    g2p = torch.mean(g2p, 1)
    cd = p2g + g2p
    return cd, p2g, g2p

def l2(p1, p2):
    diff = p1-p2
    diff_2 = (torch.sum(diff*diff, dim=2))
    diff_2 = torch.mean(diff_2, dim=1)

    return diff_2

def emd(p1, p2, reg=0.1):
    uniform = torch.ones(len(p1[0]), device=p1.device) / len(p1[0])
    M = ((p1[0][None,:] - p2[0][:,None])**2).sum(2)
    return [ot.sinkhorn2(uniform, uniform, M, reg=reg, numItermax=10)]

def arcosh(x, eps=1e-5): 
    x = torch.clamp(x, min=1 + eps)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))

def calc_cd_like_hyperV2(p1, p2):
    d1, d2 = distance(p1, p2)

    d1 = arcosh(1+ 1 * d1)
    d2 = arcosh(1+ 1 * d2)


    return torch.mean(d1, 1) + torch.mean(d2, 1)

def calc_cd_like_InfoV2(p1, p2):

    dist1, dist2 = distance(p1, p2)
    dist1 = torch.clamp(dist1, min=1e-9)
    dist2 = torch.clamp(dist2, min=1e-9)
    d1 = torch.sqrt(dist1)
    d2 = torch.sqrt(dist2)

    distances1 = - torch.log(torch.exp(-0.5* d1)/(torch.sum(torch.exp(-0.5* d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
    distances2 = - torch.log(torch.exp(-0.5 * d2)/(torch.sum(torch.exp(-0.5* d2) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)

    return (torch.sum(distances1, 1) + torch.sum(distances2, 1)) / 2

class TrainLoss(BasicLoss):
    def __init__(self, args):
        super().__init__()
        self.loss_name = ['loss_g', 'loss_d', \
            'T1_partial_cost', 'T1_complete_cost', 'T2_partial_cost', 'T2_complete_cost', \
            'D1_partial_loss', 'D1_complete_loss', 'D2_partial_loss', 'D2_complete_loss', 'D_real_loss', 'grad_penalty_real','grad_penalty_fake']
        self.loss_num = len(self.loss_name)
        self.phi1 = select_phi(args.phi1, args.alpha)
        self.phi2 = select_phi(args.phi2, args.alpha)        
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.cd_idx1, self.cd_idx2 = self.type2idx(args.cost_type)
        self.typeG = args.TC
        self.dens_coord = args.dens_coord
        self.r1_gamma = args.r1_gamma
        self.r2_gamma = args.r2_gamma
        self.cut_grad = args.cut_grad
        self.cost_function= select_cost_function(args.cost_function)
        self.wo_comple = args.wo_comple
        
    def type2idx(self, cost_type):
        if cost_type == 'uni':
            return 2, 2
        elif cost_type == 'bi':
            return 0, 0
        elif cost_type == 'mix':
            return 2, 0
        else:
            raise NotImplementedError

    def cost(self, samples):
        input_partial, input_complete, T1_partial_output, T1_complete_output, T2_partial_output, T2_complete_output = samples
     
        T1_partial_cost = self.tau1 * self.cost_function(T1_partial_output, input_partial)
        T1_complete_cost = self.tau1 * self.cost_function(T1_complete_output, input_complete)
        

        if T2_complete_output is not None:
            if self.cut_grad:
                T2_partial_cost = self.tau2 * self.cost_function(T2_partial_output, T1_partial_output.detach())
                T2_complete_cost = self.tau2 * self.cost_function(T2_complete_output, input_complete.detach())
            else:
                T2_partial_cost = self.tau2 * self.cost_function(T2_partial_output, input_partial)
                T2_complete_cost = self.tau2 * self.cost_function(T2_complete_output, input_complete)
        
            density_loss1 = self.dens_coord * (density_loss(T2_partial_output) + density_loss(T2_complete_output))
        
        else:
            T2_partial_cost, T2_complete_cost = torch.zeros_like(T1_partial_cost), torch.zeros_like(T1_complete_cost)
        
            density_loss1 = self.dens_coord * (density_loss(T1_partial_output) + density_loss(T1_complete_output))

        return T1_partial_cost, T1_complete_cost, T2_partial_cost, T2_complete_cost, density_loss1
    
    def batch_forward(self, samples, values):
        _, input_complete, _, _, T2_partial_output, T2_complete_output = samples
        V1_partial, V1_complete, V2_partial, V2_complete, V_real = values
        T1_partial_cost, T1_complete_cost, T2_partial_cost, T2_complete_cost, density_loss1 = self.cost(samples)
        
       
       
        loss_g = (T2_partial_cost + T2_complete_cost) \
                 - V2_partial - V2_complete \
                + density_loss1
 

        D1_partial_loss = self.phi1(V1_partial - T1_partial_cost)
        D1_complete_loss = self.phi1(V1_complete - T1_complete_cost)
        D2_partial_loss = self.phi1(V2_partial - T2_partial_cost)
        D2_complete_loss = self.phi1(V2_complete - T2_complete_cost)
        
        if self.typeG in ['T1','T2']:
            D_real_loss = 2 * self.phi2(-V_real)

        else:
            D_real_loss = 2 * self.phi2(-V_real)
    
        
        ## R1 regularization
        if self.r1_gamma != 0:
            grad_real = torch.autograd.grad(outputs=V_real.sum(), inputs=input_complete, create_graph=True)[0].contiguous()
            grad_penalty_real = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2)
            grad_penalty_real = self.r1_gamma / 2 * grad_penalty_real
        else:
            grad_penalty_real = torch.zeros_like(D_real_loss)

        ## R2 regularization
        if self.r2_gamma !=0:
            grad_fake_0 = torch.autograd.grad(outputs=V2_partial.sum(), inputs=T2_partial_output, create_graph=True)[0].contiguous()
            grad_penalty_fake0 = (grad_fake_0.view(grad_fake_0.shape[0], -1).norm(2, dim=1) ** 2)

            grad_fake_1 = torch.autograd.grad(outputs=V2_complete.sum(), inputs=T2_complete_output, create_graph=True)[0].contiguous()
            grad_penalty_fake1 = (grad_fake_1.view(grad_fake_1.shape[0], -1).norm(2, dim=1) ** 2)

            grad_penalty_fake = self.r2_gamma / 2 * (grad_penalty_fake0 + grad_penalty_fake1) / 2

        else:
            grad_penalty_fake = torch.zeros_like(D_real_loss)

    
        loss_d = D2_partial_loss + D2_complete_loss + D_real_loss + grad_penalty_real + (grad_penalty_fake) 
     

        return [loss_g, loss_d, \
            T1_partial_cost, T1_complete_cost, T2_partial_cost, T2_complete_cost, \
            D1_partial_loss, D1_complete_loss, D2_partial_loss, D2_complete_loss, D_real_loss, grad_penalty_real, grad_penalty_fake]

class TestLoss(BasicLoss):
    def __init__(self, args):
        super().__init__()
        self.loss_name = ['T2_CD', 'T1_CD', 'T2_F0.1', 'T1_F0.1', 'T2_F1', 'T1_F1', 'T2_Density', 'T1_Density']
        self.loss_num = len(self.loss_name)
        #self.typeG = args.TC
        self.distance = ChamferDistance()

    def cd1(self, p1, p2):
        p2g, g2p = self.distance(p1, p2)
        p2g = torch.sqrt(p2g)
        g2p = torch.sqrt(g2p)
        p2g = torch.mean(p2g, 1)
        g2p = torch.mean(g2p, 1)
        cd = (p2g + g2p)/2
        return cd
    
    def density_loss(self, x):
        x1 = x.unsqueeze(1)
        x2 = x.unsqueeze(2)
        diff = (x1-x2).norm(dim=-1)
        diff, idx = diff.topk(16, largest=False)
        loss = diff[:,:,1:].mean(2)
        mean = loss.mean(1)
        loss = loss.std(1)
        return loss, mean
        
    def batch_forward(self, samples, data):
        input_partial, input_complete, T1_partial_output, T1_complete_output, T2_partial_output, T2_complete_output = samples
        gt = data[0][1]

        if T2_partial_output is None:
            T2_partial_output = T1_partial_output

        T2_cd = self.cd1(T2_partial_output, gt)
        T1_cd = self.cd1(T1_partial_output, gt)
        
        T2_fcd_0p001 = calc_fcd(T2_partial_output, gt, a=0.001)
        T2_fcd_0p01 = calc_fcd(T2_partial_output, gt, a=0.01)
        T1_fcd_0p001 = calc_fcd(T1_partial_output, gt, a=0.001)
        T1_fcd_0p01 = calc_fcd(T1_partial_output, gt, a=0.01)
        
        T2_den_loss, mean = self.density_loss(T2_partial_output)
        T1_den_loss, mean = self.density_loss(T1_partial_output)

        return [T2_cd, T1_cd, T2_fcd_0p001, T1_fcd_0p001, T2_fcd_0p01, T1_fcd_0p01, T2_den_loss, T1_den_loss]    
    
