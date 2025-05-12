import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, LogNormal, MixtureSameFamily, Pareto
from tqdm import tqdm
import math
import pdb



# ----------------------------------------------------------------------------
# Physical reference scales (nondimensionalization)
# ----------------------------------------------------------------------------
E_ref, Gc_ref, strength_ref, rho_ref = 1e12, 1e4, 1e9, 3000
L_phys, sr_phys_ref = 0.05, 1e7

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class MultiFileDataset(Dataset):
    def __init__(self, file_paths, material_list, n_bins=50, damage_threshold=0.5):
        self.damage_threshold = damage_threshold
        self.entries = []
        for path in file_paths:
            f = h5py.File(path, 'r')
            for mat in material_list:
                grp      = f[mat]
                sr_phys  = float(grp.attrs['strain_rate'])
                sr_nd    = sr_phys / sr_phys_ref
                Nsim     = grp['damage'].shape[0]
                E_arr    = grp['E_samples'][:]
                Gc_arr   = grp['Gc_samples'][:]
                str_arr  = grp['strength_samples'][:]
                rho_arr  = grp['density_samples'][:]
                for i in range(Nsim):
                    p = torch.tensor([
                        E_arr[i]/E_ref,
                        Gc_arr[i]/Gc_ref,
                        str_arr[i]/strength_ref,
                        rho_arr[i]/rho_ref,
                        sr_nd
                    ], dtype=torch.float32)
                    self.entries.append((grp, i, p, sr_phys))

        # spatial grid (normalized 0–1)
        self.Nnodes       = self.entries[0][0]['damage'].shape[2]
        self.x_grid  = torch.linspace(0,1,self.Nnodes, dtype=torch.float32).unsqueeze(-1)
        

        # prepare fragment‐size histogram bins in *physical* units
        self.dx_nd   = 1.0 / (self.Nnodes - 1)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        grp, i, p, sr_phys = self.entries[idx]
        # final snapshot indices
        dlen = int(grp['damage_lengths'][i]) - 1
        slen = int(grp['stress_lengths'][i]) - 1
        vlen = int(grp['velocity_lengths'][i]) - 1

        # pull out fields (shape: Nnodes)
        D = torch.from_numpy(grp['damage'][i, dlen, :].astype(np.float32))
        S = torch.from_numpy(grp['stress'][i, slen, :].astype(np.float32))
        V = torch.from_numpy(grp['velocity'][i, vlen, :].astype(np.float32))
        mask = (D < self.damage_threshold)
        frag_sizes = []
        start = 0
        while start < self.Nnodes:
            if mask[start]:
                end = start
                while end < self.Nnodes and mask[end]:
                    end += 1
                length_nd = (end - start) * self.dx_nd
                frag_sizes.append(length_nd)
                start = end
            else:
                start += 1
        
        frag_sizes = torch.tensor(frag_sizes, dtype=torch.float32)
        N_frags    = torch.tensor(frag_sizes.size(0), dtype=torch.int64)
        V_tot      = torch.tensor(1.0, dtype=torch.float32)
        # # 1) compute fragment sizes (distance between crack‐nodes)
        # idxs = np.where(D.numpy() > 0.5)[0]
        # if len(idxs) >= 2:
        #     # node‐spacing in physical meters
        #     dx_phys = L_phys / (D.shape[0] - 1)
        #     # differences in node‐indices → physical lengths
        #     sizes = np.diff(idxs) * dx_phys
        # else:
        #     # if fewer than two cracks, treat whole bar as one fragment
        #     sizes = np.array([L_phys], dtype=np.float32)

        # # 2) histogram & CDF
        # counts, edges = np.histogram(sizes, bins=self.bin_edges)
        # cdf = np.cumsum(counts).astype(np.float32)
        # if cdf[-1] > 0:
        #     cdf /= cdf[-1]

        # convert to torch
        # bins    = torch.from_numpy(edges[1:].astype(np.float32))  # upper edges, shape (n_bins,)
        # cdf_gt  = torch.from_numpy(cdf)                           # shape (n_bins,)

        return (
            p, 
            self.x_grid.clone(), 
            D, S, V, 
            torch.tensor(sr_phys, dtype=torch.float32),
            frag_sizes,
            N_frags,
            V_tot
        )

def fracture_collate(batch):
    # batch is a list of tuples (p, x, D, S, V, sr_phys, frag_sizes)
    ps, xs, Ds, Ss, Vs, srs, frags, N_frags_list, V_tot_list = zip(*batch)
    return (
        torch.stack(ps, 0),
        torch.stack(xs, 0),
        torch.stack(Ds, 0),
        torch.stack(Ss, 0),
        torch.stack(Vs, 0),
        torch.stack(srs, 0),
        list(frags),    # leave as a list of 1D tensors
        torch.stack(N_frags_list, dim=0),
        torch.stack(V_tot_list, dim=0)
    )


class UNet1d(nn.Module):
    def __init__(self, in_ch=3, base_width=64, depth=4, ksize=3):
        super().__init__()
        pad = ksize // 2

        # Down‐path
        self.downs = nn.ModuleList()
        ch = in_ch
        w  = base_width
        for _ in range(depth):
            self.downs.append(nn.Sequential(
                nn.Conv1d(ch,    w, ksize, padding=pad),
                nn.GELU(),
                nn.Conv1d(w,    w, ksize, padding=pad),
                nn.GELU()
            ))
            ch = w
            w *= 2
        self.pool = nn.MaxPool1d(2, stride=2)

        # Up‐path
        self.ups = nn.ModuleList()
        for _ in range(depth):
            w //= 2
            deconv = nn.ConvTranspose1d(ch, w, kernel_size=2, stride=2)
            convs  = nn.Sequential(
                nn.Conv1d(w*2, w, ksize, padding=pad),
                nn.GELU(),
                nn.Conv1d(w,   w, ksize, padding=pad),
                nn.GELU()
            )
            self.ups.append(nn.ModuleList([deconv, convs]))
            ch = w

        # final map to 3 fields
        self.out = nn.Conv1d(ch, 3, kernel_size=1)

    def forward(self, x):
        # x: (B,3,N)
        skips = []
        h = x
        for down in self.downs:
            h = down(h)
            skips.append(h)
            h = self.pool(h)
        for (deconv, convs), skip in zip(self.ups, reversed(skips)):
            h = deconv(h)
            h = torch.cat([h, skip], dim=1)
            h = convs(h)
        h_feat = h                    # <-- save this for MDN
        out    = self.out(h)          # (B,3,N)
        return out, h_feat            # return both


class FractureUNet(nn.Module):
    def __init__(self,
                 in_ch: int = 7,           # 5 + p + t
                 base_width: int = 64,
                 depth: int = 4,
                 ksize: int = 3,
                 mdn_components: int = 16):
        super().__init__()
        # 1) U-Net that returns (unet_out, feature_map)
        self.unet = UNet1d(in_ch=in_ch,
                           base_width=base_width,
                           depth=depth,
                           ksize=ksize)
        
        # 2) probabilistic head (MDN) on the feature map
        self.K   = mdn_components
        self.mdn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),         # → (B, base_width, 1)
            nn.Flatten(1),                   # → (B, base_width)
            nn.Linear(base_width, 3*self.K)
        )
        # initialize MDN linear gently
        lin = self.mdn[2]
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)

    def forward(self, p, x, t):
        B, N, _ = x.shape
        # build U-Net input
        p_grid = p.unsqueeze(-1).repeat(1,1,N)
        xg, tg = x.permute(0,2,1), t.permute(0,2,1)
        inp    = torch.cat([p_grid, xg, tg], dim=1)  # (B, in_ch, N)

        # 1) decode → deterministic prediction + features
        unet_out, feat = self.unet(inp)
        # unet_out is (B, 3, N):  D_logit, S, v
        D_logit = unet_out[:,0,:]
        S       = unet_out[:,1,:]
        v       = unet_out[:,2,:]

        # 2) MDN head on the same features
        h_mdn    = self.mdn(feat)            # (B, 3*K)
        logits, mus, logsig = h_mdn.chunk(3, dim=1)

        # clamp + floor scales for stability
        logsig = logsig.clamp(min=-5, max=5)
        # hard clamps
        logits = logits.clamp(-10, 10)
        mus    = mus.clamp(-10, 10)
        logsig = logsig.clamp(-5, 5)
        sigmas = F.softplus(logsig) + 1e-3

        mixing = Categorical(logits=logits)
        comp   = LogNormal(loc=mus, scale=sigmas)
        mix    = MixtureSameFamily(mixing, comp)

        return torch.sigmoid(D_logit), S, v, mix
    
def mdn_loss(mix, frag_sizes):
        nll = 0.0
        for j, sizes in enumerate(frag_sizes):
            # 1) extract the j-th mixing logits (shape [K])
            logits_j = mix.mixture_distribution.logits[j]           # (K,)
            # 2) extract the j-th component loc/scale (each shape [K])
            comp = mix.component_distribution
            loc_j   = comp.base_dist.loc[j]                        # (K,)
            scale_j = comp.base_dist.scale[j]                      # (K,)

            # 3) rebuild a single-sample mixture distribution
            mixing_j = torch.distributions.Categorical(logits=logits_j)
            comp_j   = torch.distributions.LogNormal(loc_j, scale_j)
            single_mix = torch.distributions.MixtureSameFamily(mixing_j, comp_j)
            
            logps = single_mix.log_prob(sizes)      # shape [Nj]
            nll  += -logps.mean()                   # mean over Nj
        return nll

def mass_conservation_loss(mix, N_fragments, V_total):
    """
    mix: MixtureSameFamily over sizes X
    N_fragments: (B,) number of fragments
    V_total:      (B,) known total volume (or length) of bar
    """
    # E[X] per batch
    EX = mix.component_distribution.mean * mix.mixture_distribution.probs
    EX = EX.sum(-1)      # (B,)
    V_pred = EX * N_fragments
    # pdb.set_trace()
    return F.mse_loss(V_pred, V_total)
# ----------------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------------
def train():
    file_paths = glob.glob('1_D_Fracture/fracture_results_h5/*.h5')
    mats = ["Aluminum Alloy","Titanium Alloy","CFRP","Stainless Steel",
            "Beryllium","Glass","Copper Alloy"]
    ds = MultiFileDataset(file_paths,mats)
    n= len(ds); nt=int(0.8*n)
    train_ds,val_ds = random_split(ds,[nt,n-nt])
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=fracture_collate
    )
    val_loader   = DataLoader(val_ds,  batch_size=1,shuffle=False,num_workers=2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    components = 32
    ksize = 3
    width = 64
    depth = 4
    model = FractureUNet(in_ch=5+1+1, base_width=width, depth=depth, ksize=ksize, mdn_components=components).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-6
    )
    writer = SummaryWriter(f"runs/Fracture_MDN_UNet_width_{width}_components_{components}_ksize_{ksize}_depth_{depth}")
    epochs=50
    global_step = 0
    beta_final = 1
    phys_warmup = 10
    phys_final = 1e-4
    N = 50000
    lambda_mass = 10.0
    for ep in tqdm(range(epochs)):
        total_loss = 0.0    
        beta = min(1.0, ep/10) * beta_final
        # beta = beta_final
        # lambda_phys = phys_final * max(0.0, min(1.0, (ep - phys_warmup)/phys_warmup))
        lambda_phys = phys_final

        for p, x, D_gt, S_gt, V_gt, sr_phys, frag_sizes, N_frags, V_tot in tqdm(train_loader):
                    # Move to device
            p, x = p.to(device), x.to(device)
            D_gt, S_gt, V_gt = [t.to(device) for t in (D_gt, S_gt, V_gt)]
            t0 = torch.zeros_like(x)     # initial time
            t1 = torch.ones_like(x)
            N_frags = N_frags.to(device)
            V_tot   = V_tot.to(device)
            frag_sizes = [sz.to(device) for sz in frag_sizes]


            D0, S0, v0, mix0 = model(p, x, t0)
            D1, S1, v1, mix = model(p, x, t1)
   
            
            nll = mdn_loss(mix, frag_sizes)
            loss_mdn = beta * nll
            
            #mass conservation loss
            N_frags = torch.tensor([len(sz) for sz in frag_sizes], device=device, dtype=torch.float32)
            l_mass = mass_conservation_loss(mix, N_frags, V_tot)

            D_t = D1 - D0    # (B,N)
            v_t = v1 - v0    # (B,N)

            # l_mass = mass_conservation_loss(mix, N_frags, V_tot)
            # pdb.set_trace()
           # ----- discrete‐PINN physics loss -----
            # finite‐difference for ∂ₓS and ∂ₓv
            dx       = 1.0/(N-1)
            kernel = torch.tensor([-1.,0.,1.], device=device).view(1,1,3) / (2*dx)
            S_x    = F.conv1d(S1.unsqueeze(1), kernel, padding=1).squeeze(1)  # (B,N)
            v_x    = F.conv1d(v1.unsqueeze(1), kernel, padding=1).squeeze(1)  # (B,N)


            res_mom  = v_t - S_x
            res_dam  = D_t - (1.0 - D_t) * v_x
            loss_mom = res_mom.pow(2).mean()
            loss_dam = res_dam.pow(2).mean()
            loss_phys= loss_mom + loss_dam

            # 4) *True* IC‐loss at t=0:
            #   target: D0=0, S0_nd=1, v0_nd = x-0.5
            x_flat   = x.squeeze(-1)      # (B,N)
            v0_tar   = x_flat - 0.5       # (B,N)
            S0_tar = 0.99 * torch.ones_like(S0)
            loss_ic  = (D0**2 + (S0 - S0_tar)**2 + (v0 - v0_tar)**2).mean()

            # 5) *True* BC‐loss for velocity at t=1:
            #   v1[:,0]→-0.5,  v1[:,-1]→+0.5
            vL = v1[:, 0]   # left boundary
            vR = v1[:, -1]  # right boundary
            loss_bc = ((vL + 0.5)**2 + (vR - 0.5)**2).mean()



            # ----- total loss -----
            loss = loss_mdn + lambda_phys * loss_phys + lambda_phys * loss_ic + lambda_phys * loss_bc + l_mass * lambda_mass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            

            total_loss += loss.item()
            global_step += 1
            if global_step % 1 == 0:
                writer.add_scalar('Loss_VAE_MDN/train', loss.item(), global_step)
                writer.add_scalar('Loss_VAE_MDN/loss_phys', loss_phys.item(), global_step)
                writer.add_scalar('Loss_VAE_MDN/loss_mom', loss_mom.item(), global_step)
                writer.add_scalar('Loss_VAE_MDN/loss_dam', loss_dam.item(), global_step)
                writer.add_scalar('Loss_VAE_MDN/loss_mdn', loss_mdn.item(), global_step)
                writer.add_scalar('Loss_VAE_MDN/loss_ic', loss_ic.item(), global_step)
                writer.add_scalar('Loss_VAE_MDN/loss_bc', loss_bc.item(), global_step)
                writer.add_scalar('Loss_VAE_MDN/loss_mass', l_mass.item(), global_step)
                
        print(f"Epoch {ep}, Train Loss {total_loss/len(train_loader):.4e}")
        # validation omitted for brevity
        
    writer.close()
    torch.save(model.state_dict(), f"Fracture_MDN_UNet_width_{width}_components_{components}_ksize_{ksize}_depth_{depth}")

if __name__=='__main__':
    train()
