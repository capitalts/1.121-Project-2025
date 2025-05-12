import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, LogNormal, MixtureSameFamily, Pareto
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pdb
import math
from torch.distributions import Weibull

bce_logits = nn.BCEWithLogitsLoss()
# ----------------------------------------------------------------------------
# 0) Physical reference scales
# ----------------------------------------------------------------------------
E_ref        = 1e12    # Pa
Gc_ref       = 1e5     # J/m²
strength_ref = 1e10   # Pa
rho_ref      = 10000    # kg/m³
L_phys       = 0.05    # m
sr_ref = 1e7

# ----------------------------------------------------------------------------
# 1) Dataset: returns (params, x, sig, v, D) at t=1
# ----------------------------------------------------------------------------
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiFileFragmentDataset(Dataset):
    """
    Dataset that returns for each simulation:
      - p: nondimensional parameter vector (5,)
      - frag_sizes: 1D Tensor of fragment lengths (normalized 0-1)
      - N_frags: Tensor count of fragments per sample
      - V_tot: Tensor total normalized length (1.0)

    Provides `collate_fn` to batch variable-length fragments.
    """
    def __init__(self, file_paths, material_list, L_phys=0.05, damage_threshold=0.5):
        self.entries = []
        self.material_list = material_list
        self.L_phys = L_phys
        self.damage_threshold = damage_threshold

        for path in file_paths:
            f = h5py.File(path, 'r')
            for mat in material_list:
                grp = f[mat]
                sr_phys = float(grp.attrs['strain_rate'])
                Nsim = grp['damage'].shape[0]

                E_arr        = grp['E_samples'][:]
                Gc_arr       = grp['Gc_samples'][:]
                strength_arr = grp['strength_samples'][:]
                rho_arr      = grp['density_samples'][:]

                for i in range(Nsim):
                    E_nd        = float(E_arr[i])      / E_ref
                    Gc_nd       = float(Gc_arr[i])     / Gc_ref
                    strength_nd = float(strength_arr[i])/ strength_ref
                    rho_nd      = float(rho_arr[i])    / rho_ref

                    c_char = np.sqrt(E_arr[i] / rho_arr[i])
                    tau    = L_phys / c_char
                    sr_nd  = sr_phys / sr_ref 
                    # sr_nd_log = sr_nd_log / 8.0           # scale into roughly [0,1]

                    self.entries.append({
                        'grp':     grp,
                        'idx':     i,
                        'params':  torch.tensor([
                            E_nd, Gc_nd, strength_nd, rho_nd, sr_nd
                        ], dtype=torch.float32),
                    })

        first = self.entries[0]
        self.Nnodes = first['grp']['damage'].shape[2]
        self.dx_nd   = 1.0 / (self.Nnodes - 1)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e   = self.entries[idx]
        grp = e['grp']
        i   = e['idx']
        material = grp.name.split('/')[-1]
        dlen = int(grp['damage_lengths'][i])
        D = grp['damage'][i, dlen-1, :].astype(np.float32)

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

        return e['params'], frag_sizes, N_frags, V_tot, material

    @staticmethod
    def collate_fn(batch):
        """
        Collate for DataLoader:
          - params: stack to (B,5)
          - frag_sizes: list of 1D Tensors
          - N_frags: stack to (B,)
          - V_tot: stack to (B,)
        """
        params, frag_lists, N_frags_list, V_tot_list = zip(*batch)
        p_batch        = torch.stack(params, dim=0)
        frag_sizes_batch = [fs for fs in frag_lists]
        N_frags_batch  = torch.stack(N_frags_list, dim=0)
        V_tot_batch    = torch.stack(V_tot_list, dim=0)
        return p_batch, frag_sizes_batch, N_frags_batch, V_tot_batch


class FragmentMDN(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=128, n_components=5):
        super().__init__()
        self.n_comp = n_components
        
        # Shared MLP backbone
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Heads for mixture weights, means, and log‐stds
        self.logits = nn.Linear(hidden_dim, n_components)      # unnormalized α
        self.means  = nn.Linear(hidden_dim, n_components)      # μ in log‐space
        self.log_stds = nn.Linear(hidden_dim, n_components)    # log σ > 0

    def forward(self, p):
        """
        p: (B,5) nondimensional material+strain‐rate parameters
        returns: a torch.distributions.MixtureSameFamily over fragment sizes
        """
        h = self.net(p)                           # (B, hidden_dim)
        
        logit_a = self.logits(h)                  # (B, n_comp)
        mixing = Categorical(logits=logit_a)      # π_k
        
        μ       = self.means(h)                   # (B, n_comp)
        σ       = F.softplus(self.log_stds(h))    # ensure >0
        
        # Create a LogNormal for each comp on each batch element
        comp_dist = LogNormal(loc=μ, scale=σ)     # event_shape=(n_comp,)
        
        # Build mixture
        mix = MixtureSameFamily(mixing, comp_dist)
        return mix
    
class WeibullMDN(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=128, n_components=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.logits    = nn.Linear(hidden_dim, n_components)
        self.log_scales = nn.Linear(hidden_dim, n_components)  # λ > 0
        self.shapes     = nn.Linear(hidden_dim, n_components)  # k > 0

    def forward(self, x):
        h    = self.net(x)
        π    = F.softmax(self.logits(h), dim=-1)
        λ    = F.softplus(self.log_scales(h)) + 1e-6
        k    = F.softplus(self.shapes(h))     + 1e-3
        mix  = MixtureSameFamily(
            Categorical(probs=π),
            Weibull(scale=λ, concentration=k)
        )
        return mix

# --- 1) Pure Pareto MDN ---
class ParetoMDN(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=128, n_components=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.logits    = nn.Linear(hidden_dim, n_components)
        self.log_xm    = nn.Linear(hidden_dim, n_components)
        self.log_alpha = nn.Linear(hidden_dim, n_components)

    def forward(self, p):
        h      = self.net(p)                           # (B, hidden_dim)
        logits = self.logits(h)                        # (B, K)
        probs  = F.softmax(logits, dim=-1)             # mixture weights

        xm_raw    = self.log_xm(h)                     # (B, K)
        alpha_raw = self.log_alpha(h)                  # (B, K)
        xm    = F.softplus(xm_raw)    + 1e-6           # scale >0
        alpha = F.softplus(alpha_raw)  + 1e-3           # shape >0

        # Mixture of Pareto components
        comp = Pareto(scale=xm, alpha=alpha)
        mix  = MixtureSameFamily(Categorical(probs=probs), comp)
        return mix
    
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


def tail_exponent_loss(mix, τ_phys, x_tail):
    """
    Enforce p(x) ~ x^{-τ_phys} at large x.
    We approximate the local slope of log p(x) near x_tail.
    """
    # compute log-prob at x and at 1.1*x
    logp1 = mix.log_prob(x_tail)
    logp2 = mix.log_prob(x_tail * 1.1)
    # finite‐difference exponent:  d log p / d log x ≈ (logp2 - logp1)/(log(1.1))
    τ_pred = - (logp2 - logp1) / math.log(1.1)
    # penalize deviation from τ_phys
    return F.mse_loss(τ_pred, τ_phys)

if __name__ == '__main__':
    file_paths    = glob.glob('1_D_Fracture/fracture_results_h5/*.h5')
    material_list = ["Aluminum Alloy","Titanium Alloy","CFRP","Stainless Steel",
                     "Beryllium","Glass","Copper Alloy"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = MultiFileFragmentDataset(file_paths, material_list)
    
    λ_mass, λ_tail = 10.0, 0
    mse_loss  = nn.MSELoss()
    kl_loss   = nn.KLDivLoss(reduction="batchmean")
    batch_size = 8
    hidden_dim = 128
    n_components = 32
    trial_name = f"MDN_Fracture_hidden_dim_{hidden_dim}_components_{n_components}_batch_size_{batch_size}_mass_{λ_mass}_tail_{λ_tail}"

    writer = SummaryWriter(trial_name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=MultiFileFragmentDataset.collate_fn)
    global_step = 0
    # ----------------------------
    # Example training loop
    # ----------------------------
    model = FragmentMDN(in_dim=5, hidden_dim=hidden_dim, n_components=n_components).to(device)
    # model = WeibullMDN(in_dim=5, hidden_dim=128, n_components=32).to(device)
    # model = ParetoMDN(in_dim=5, hidden_dim=128, n_components=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)

    
    τ_phys = torch.tensor([2.5], device=device)         # your expected power‐law exponent
    x_tail = torch.tensor([0.9], device=device)        # evaluate tail at 90th percentile (normalized)
    for epoch in tqdm(range(100)):
        for p, frag_sizes, N_frags, V_tot, _ in loader:
            """
            - p:          (B,5) input parameters
            - frag_sizes: list of length B, each a 1D tensor [x1,x2,...,xM]
            - N_frags:    (B,) number of fragments in each sample
            - V_tot:      (B,) total volume/length of bar
            """
            p = p.to(device)
            frag_sizes = [sz.to(device) for sz in frag_sizes]
            N_frags = N_frags.to(device)
            V_tot   = V_tot.to(device)
            # pdb.set_trace() 
            # 1) build mixture
            mix = model(p)   # MixtureSameFamily
            # pdb.set_trace()
            # 2) data NLL loss
            # sum log‐probs of each true fragment size under the mixture
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

            # nll = 0.0
            # for j, sizes in enumerate(frag_sizes):
            #     # sizes: 1-D tensor of length Nj

            #     # 1) extract the j-th mixture weights (shape [K])
            #     probs_j = mix.mixture_distribution.probs[j]        # (K,)

            #     # 2) extract the j-th Weibull parameters (each shape [K])
            #     wd = mix.component_distribution                   # a Weibull with batch_shape=[B,K]
            #     scale_j        = wd.scale[j]                      # (K,)
            #     concentration_j= wd.concentration[j]              # (K,)

            #     # 3) rebuild a single-sample mixture of Weibulls
            #     mixing_j = torch.distributions.Categorical(probs=probs_j)
            #     comp_j   = torch.distributions.Weibull(
            #                 scale=scale_j,
            #                 concentration=concentration_j
            #             )
            #     single_mix = torch.distributions.MixtureSameFamily(mixing_j, comp_j)

            #     # 4) compute mean-per-fragment negative log-likelihood
            #     logps = single_mix.log_prob(sizes.to(device))     # (Nj,)
            #     nll  += -logps.mean()

            # 5) average over the batch
            nll = nll / len(frag_sizes)

            # 3) physics‐informed penalties
            l_mass = mass_conservation_loss(mix, N_frags, V_tot)
            l_tail = tail_exponent_loss(mix, τ_phys, x_tail)

            loss = nll + λ_mass * l_mass + λ_tail * l_tail
            
            # 4) backward & step
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 5) logging
            # print(f"nll={nll.item():.3f}  mass={l_mass.item():.3e}  tail={l_tail.item():.3e}")
            if global_step % 100 == 0:
                writer.add_scalar('loss/nll', nll.item(), global_step)
                writer.add_scalar('loss/mass', l_mass.item(), global_step)
                writer.add_scalar('loss/tail', l_tail.item(), global_step)
                writer.add_scalar('loss/total', loss.item(), global_step)
            global_step += 1

    # Save the model
    torch.save(model.state_dict(), f'{trial_name}.pth')
    writer.close()
    print(f"Model saved as {trial_name}.pth")