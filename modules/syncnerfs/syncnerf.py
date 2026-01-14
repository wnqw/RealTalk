import torch
import torch.nn as nn
import torch.nn.functional as F
from .renderer import SyncNeRFRenderer
from .encoders.encoding import get_encoder
from modules.radnerfs.cond_encoder import AudioNet, AudioAttNet, MLP

class SyncNeRF(SyncNeRFRenderer):
    def __init__(self,
                 opt,
                 cond_out_dim = 64,
                 # torso net (hard coded for now)
                 ):
        super().__init__(opt)
        
        self.cond_in_dim = 68*3
        self.cond_out_dim = opt.cond_out_dim // 2 * 2
        self.cond_win_size = opt.cond_win_size
        self.smo_win_size = opt.smo_win_size     
        self.cond_prenet = AudioNet(self.cond_in_dim, self.cond_out_dim, win_size=self.cond_win_size)
        self.att = self.opt.att
        if self.att > 0:
            self.cond_att_net = AudioAttNet(self.cond_out_dim, seq_len=self.smo_win_size)
    

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 64
        self.geo_feat_dim = 64
        if self.opt.au45:
            self.eye_att_net = MLP(self.in_dim, 1, 16, 2)
            self.eye_dim = 1 if self.exp_eye else 0
        else:
            if self.opt.bs_area == "upper":
                self.eye_att_net = MLP(self.in_dim, 7, 64, 2)
                self.eye_dim = 7 if self.exp_eye else 0
            elif self.opt.bs_area == "single":
                self.eye_att_net = MLP(self.in_dim, 4, 64, 2)
                self.eye_dim = 4 if self.exp_eye else 0
            elif self.opt.bs_area == "eye":
                self.eye_att_net = MLP(self.in_dim, 2, 64, 2)
                self.eye_dim = 2 if self.exp_eye else 0
        self.sigma_net = MLP(self.in_dim + self.cond_out_dim + self.eye_dim, 1 + self.geo_feat_dim, self.hidden_dim, self.num_layers)
        ## color network
        self.num_layers_color = 2
        self.hidden_dim_color = 64
        self.encoder_dir, self.in_dim_dir = get_encoder('spherical_harmonics')
        self.color_net = MLP(self.in_dim_dir + self.geo_feat_dim + self.individual_dim, 3, self.hidden_dim_color, self.num_layers_color)

        self.unc_net = MLP(self.in_dim, 1, 32, 2)

        self.aud_ch_att_net = MLP(self.in_dim, self.cond_out_dim, 64, 2)

        self.testing = False

        if self.torso:
            # torso deform network
            self.register_parameter('anchor_points', 
                                    nn.Parameter(torch.tensor([[0.01, 0.01, 0.1, 1], [-0.1, -0.1, 0.1, 1], [0.1, -0.1, 0.1, 1]])))
            self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('frequency', input_dim=2, multires=8)
            # self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=1, base_resolution=16, log2_hashmap_size=16, desired_resolution=512)
            self.anchor_encoder, self.anchor_in_dim = get_encoder('frequency', input_dim=6, multires=3)
            self.torso_deform_net = MLP(self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso, 2, 32, 3)

            # torso color network
            self.torso_encoder, self.torso_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=16, desired_resolution=2048)
            self.torso_net = MLP(self.torso_in_dim + self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso, 4, 32, 3)


    def forward_torso(self, x, poses, c=None):
        # x: [N, 2] in [-1, 1]
        # head poses: [1, 4, 4]
        # c: [1, ind_dim], individual code

        # test: shrink x
        x = x * self.opt.torso_shrink

        # deformation-based
        wrapped_anchor = self.anchor_points[None, ...] @ poses.permute(0, 2, 1).inverse()
        wrapped_anchor = (wrapped_anchor[:, :, :2] / wrapped_anchor[:, :, 3, None] / wrapped_anchor[:, :, 2, None]).view(1, -1)
        # print(wrapped_anchor)
        # enc_pose = self.pose_encoder(poses)
        enc_anchor = self.anchor_encoder(wrapped_anchor)
        enc_x = self.torso_deform_encoder(x)

        if c is not None:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1), c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1)], dim=-1)

        dx = self.torso_deform_net(h)
        
        x = (x + dx).clamp(-1, 1)

        x = self.torso_encoder(x, bound=1)

        # h = torch.cat([x, h, enc_a.repeat(x.shape[0], 1)], dim=-1)
        h = torch.cat([x, h], dim=-1)

        h = self.torso_net(h)

        alpha = torch.sigmoid(h[..., :1])*(1 + 2*0.001) - 0.001
        color = torch.sigmoid(h[..., 1:])*(1 + 2*0.001) - 0.001

        return alpha, color, dx


    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    

    def cal_cond_feat(self, cond):
        # 3,1,204
        cond_feat = self.cond_prenet(cond) # 3,64

        if self.att > 0:
            cond_feat = self.cond_att_net(cond_feat) # [1, 64] 
        return cond_feat

    
    def predict_uncertainty(self, unc_inp):
        if self.testing or not self.opt.unc_loss:
            unc = torch.zeros_like(unc_inp)
        else:
            unc = self.unc_net(unc_inp.detach())

        return unc


    def forward(self, x, d, enc_a, c, e=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # enc_a: [1, aud_dim]
        # c: [1, ind_dim], individual code
        # e: [1, 1], eye feature
        enc_x = self.encode_x(x, bound=self.bound)

        sigma_result = self.density(x, enc_a, e, enc_x)
        sigma = sigma_result['sigma']
        geo_feat = sigma_result['geo_feat']
        aud_ch_att = sigma_result['ambient_aud']
        eye_att = sigma_result['ambient_eye']

        # color
        enc_d = self.encoder_dir(d)

        if c is not None:
            h = torch.cat([enc_d, geo_feat, c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_d, geo_feat], dim=-1)
                
        h_color = self.color_net(h)
        color = torch.sigmoid(h_color)*(1 + 2*0.001) - 0.001
        
        uncertainty = self.predict_uncertainty(enc_x)
        uncertainty = torch.log(1 + torch.exp(uncertainty))

        return sigma, color, aud_ch_att, eye_att, uncertainty[..., None]


    def density(self, x, enc_a, e=None, enc_x=None):
        # x: [N, 3], in [-bound, bound]
        if enc_x is None:
            enc_x = self.encode_x(x, bound=self.bound)

        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att # ?

        if e is not None:
            # e = self.encoder_eye(e)
            # eye_att = torch.sigmoid(self.eye_att_net(enc_x))
            e = e.repeat(enc_x.shape[0], 1)
            eye_att = self.eye_att_net(enc_x)
            e = e * eye_att
            # e = e.repeat(enc_x.shape[0], 1)
            h = torch.cat([enc_x, enc_w, e], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w], dim=-1)

        h = self.sigma_net(h)

        sigma = torch.exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True),
        }


    def get_params(self, lr, lr_net, wd=0):

        # ONLY train torso
        if self.torso:
            params = [
                {'params': self.torso_encoder.parameters(), 'lr': lr},
                {'params': self.torso_deform_encoder.parameters(), 'lr': lr, 'weight_decay': wd},
                {'params': self.torso_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.torso_deform_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.anchor_points, 'lr': lr_net, 'weight_decay': wd}
            ]

            if self.individual_dim_torso > 0:
                params.append({'params': self.individual_codes_torso, 'lr': lr_net, 'weight_decay': wd})

            return params

        params = [
            {'params': self.cond_prenet.parameters(), 'lr': lr_net, 'weight_decay': wd}, 

            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            # {'params': self.encoder_xyz.parameters(), 'lr': lr},

            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
        ]
        if self.att > 0:
            params.append({'params': self.cond_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        if self.train_camera:
            params.append({'params': self.camera_dT, 'lr': 1e-5, 'weight_decay': 0})
            params.append({'params': self.camera_dR, 'lr': 1e-5, 'weight_decay': 0})

        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.unc_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params