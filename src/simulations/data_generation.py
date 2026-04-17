import numpy as np, matplotlib.pyplot as plt, os
from numpy.random import default_rng
rng = default_rng(0)

def gaussian2d(K,E, k0,e0, sk,se, amp=1.0):
    return amp*np.exp(-0.5*((K-k0)/sk)**2 -0.5*((E-e0)/se)**2)

def make_basis_spectra(k, e):
    K,E = np.meshgrid(k,e,indexing='ij')
    bg = 0.03 + 0.015*np.exp(-((E+0.8)/0.6)**2)
    lhb  = gaussian2d(K,E, 0.00,-0.22, 0.18,0.06, 0.9)
    lhb2 = gaussian2d(K,E, 0.28,-0.25, 0.12,0.07, 0.35)
    gap_supp = 1 - 0.95*np.exp(-0.5*((E-0.0)/0.06)**2)
    S_ccdw = (bg + lhb + lhb2)*gap_supp

    band = np.exp(-0.5*((E - (0.25*(K-0.15)))/0.07)**2) * np.exp(-0.5*((K-0.05)/0.32)**2)
    band *= 0.65
    ef = gaussian2d(K,E, 0.0,0.0, 0.22,0.04, 0.35)
    shoulder = gaussian2d(K,E, 0.0,-0.20, 0.20,0.07, 0.25)
    S_met = bg + band + ef + shoulder

    band_i = np.exp(-0.5*((E - (0.22*(K-0.10)))/0.08)**2) * np.exp(-0.5*((K-0.03)/0.34)**2)
    band_i *= 0.38
    pseudo = 1 - 0.55*np.exp(-0.5*((E-0.0)/0.07)**2)
    S_int = (bg + 0.55*lhb + band_i)*pseudo + 0.05*ef

    def norm(S):
        S = np.clip(S,0,None)
        return S/(S.max()+1e-9)
    return np.stack([norm(S_ccdw), norm(S_met), norm(S_int)], axis=0)

def affine_warp(img, k, e, theta_deg=0.0, shear=0.0):
    Nk,Ne = img.shape
    K,E = np.meshgrid(k,e,indexing='ij')
    th = np.deg2rad(theta_deg)
    A = np.array([[np.cos(th), -np.sin(th)+shear],
                  [np.sin(th),  np.cos(th)]], dtype=float)
    Ainv = np.linalg.inv(A)
    coords = np.stack([K,E],axis=-1)
    src = coords @ Ainv.T
    Ks = src[...,0]; Es = src[...,1]

    kmin,kmax = k[0], k[-1]
    emin,emax = e[0], e[-1]
    ik = (Ks-kmin)/(kmax-kmin)*(Nk-1)
    ie = (Es-emin)/(emax-emin)*(Ne-1)

    ik0 = np.floor(ik).astype(int); ie0 = np.floor(ie).astype(int)
    ik1 = ik0+1; ie1 = ie0+1
    wk = ik-ik0; we = ie-ie0

    valid = (ik0>=0)&(ik1<Nk)&(ie0>=0)&(ie1<Ne)
    ik0c = np.clip(ik0,0,Nk-1); ik1c=np.clip(ik1,0,Nk-1)
    ie0c = np.clip(ie0,0,Ne-1); ie1c=np.clip(ie1,0,Ne-1)

    v00=img[ik0c,ie0c]; v10=img[ik1c,ie0c]
    v01=img[ik0c,ie1c]; v11=img[ik1c,ie1c]
    out = (1-wk)*(1-we)*v00 + wk*(1-we)*v10 + (1-wk)*we*v01 + wk*we*v11
    out[~valid]=0.0
    return out

def make_domain_map(nx, ny, n_domains=7):
    dom = np.zeros((nx,ny),int)
    seeds = rng.uniform([0,0],[nx,ny], size=(n_domains,2))
    for i in range(nx):
        for j in range(ny):
            d = np.sum((seeds-np.array([i,j]))**2, axis=1)
            dom[i,j]=int(np.argmin(d))
    thetas = rng.choice([-12,-7,-3,0,4,9,14], size=n_domains, replace=True)
    theta = np.zeros((nx,ny),float)
    for d in range(n_domains):
        theta[dom==d]=thetas[d]
    gx = np.abs(np.diff(dom,axis=0,prepend=dom[:1,:]))
    gy = np.abs(np.diff(dom,axis=1,prepend=dom[:,:1]))
    boundary = ((gx+gy)>0).astype(float)
    for _ in range(2):
        boundary = 0.2*(np.roll(boundary,1,0)+np.roll(boundary,-1,0)+np.roll(boundary,1,1)+np.roll(boundary,-1,1)+boundary)
    boundary = (boundary-boundary.min())/(boundary.max()-boundary.min()+1e-9)
    return dom, theta, boundary

def make_pulse_masks(nx, ny):
    X,Y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny), indexing='ij')
    thA = np.deg2rad(35)
    uA =  np.cos(thA)*X + np.sin(thA)*Y
    vA = -np.sin(thA)*X + np.cos(thA)*Y
    barA = (np.abs(vA)<0.18)&(uA>-0.95)&(uA<0.95)

    thB = thA + np.deg2rad(90)
    uB =  np.cos(thB)*X + np.sin(thB)*Y
    vB = -np.sin(thB)*X + np.cos(thB)*Y
    barB = (np.abs(vB)<0.18)&(uB>-0.95)&(uB<0.95)
    return barA.astype(float), barB.astype(float)

def apply_write(w, bar, strength=0.70):
    w2=w.copy()
    a=strength*bar
    w2[...,1] += a*(1-w2[...,1])
    w2[...,0] *= (1-0.75*a)
    w2[...,2] *= (1-0.25*a)
    w2=np.clip(w2,1e-6,None)
    w2/=w2.sum(axis=-1,keepdims=True)
    return w2

def apply_erase(w, bar, boundary, pulse_dir='B', erase_strength=0.55):
    wmet=w[...,1]
    gx=np.abs(np.diff(wmet,axis=0,prepend=wmet[:1,:]))
    gy=np.abs(np.diff(wmet,axis=1,prepend=wmet[:,:1]))
    interface=gx+gy
    for _ in range(2):
        interface = 0.2*(np.roll(interface,1,0)+np.roll(interface,-1,0)+np.roll(interface,1,1)+np.roll(interface,-1,1)+interface)
    interface = interface/(interface.max()+1e-9)

    X,Y = np.meshgrid(np.linspace(0,1,w.shape[0]), np.linspace(0,1,w.shape[1]), indexing='ij')
    grid = 0.5*(np.sin(2*np.pi*6*(X if pulse_dir=='B' else Y))**2)

    hotspot = 0.65*interface + 0.55*boundary + 0.25*grid
    hotspot/=hotspot.max()+1e-9

    a=erase_strength*bar*hotspot
    w2=w.copy()
    w2[...,1] *= (1-0.85*a)
    w2[...,2] += 0.55*a*(w[...,1])
    w2[...,0] += 0.30*a*(w[...,1])
    w2=np.clip(w2,1e-6,None)
    w2/=w2.sum(axis=-1,keepdims=True)
    return w2

def energy_broaden(spec, sigma_idx):
    if sigma_idx<=0: return spec
    rad = int(max(3, np.ceil(4*sigma_idx)))
    x = np.arange(-rad,rad+1)
    ker = np.exp(-0.5*(x/sigma_idx)**2)
    ker/=ker.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v,ker,mode='same'), -1, spec)
    return out

def render_cube_fast(w, dom, theta_map, boundary_map, k, e, basis3, broaden_meV=0.0, counts=2500):
    nx,ny = w.shape[:2]
    Nk,Ne = basis3.shape[1:]
    unique_domains = np.unique(dom)
    warped = np.zeros((unique_domains.size, 3, Nk, Ne), float)
    for idx,d in enumerate(unique_domains):
        th = float(np.median(theta_map[dom==d]))
        shear = 0.06*np.tanh(2*(np.median(boundary_map[dom==d])-0.4))
        for c in range(3):
            warped[idx,c] = affine_warp(basis3[c], k, e, theta_deg=th, shear=shear)

    dom_to_idx = {int(d):i for i,d in enumerate(unique_domains)}
    didx = np.vectorize(dom_to_idx.get)(dom)

    cube = np.zeros((nx,ny,Nk,Ne), float)
    for c in range(3):
        cube += w[...,c,None,None] * warped[didx, c]

    if broaden_meV>0:
        de = (e[-1]-e[0])/(Ne-1)
        sigma_idx = (broaden_meV/1000.0)/de
        cube = energy_broaden(cube, sigma_idx)

    cube = cube/(cube.max()+1e-9)
    sigma = np.sqrt(np.clip(cube,0,None)/counts)
    noisy = cube + sigma*rng.normal(size=cube.shape)
    return np.clip(noisy,0,None)

def compute_Irat(cube, e, ef_win=(-0.02,0.02), ref_win=(-0.32,-0.18)):
    ef=(e>=ef_win[0])&(e<=ef_win[1])
    ref=(e>=ref_win[0])&(e<=ref_win[1])
    Ief=cube[...,ef].mean(axis=(-1,-2))
    Iref=cube[...,ref].mean(axis=(-1,-2))
    return Ief/(Iref+1e-9)

# --- generate ---
nx, ny = 64, 48
Nk, Ne = 72, 96
k = np.linspace(-0.6,0.6,Nk)
e = np.linspace(-0.8,0.15,Ne)

basis3 = make_basis_spectra(k,e)
dom, theta_map, boundary_map = make_domain_map(nx,ny,n_domains=7)
barA, barB = make_pulse_masks(nx,ny)

w0=np.zeros((nx,ny,3),float)
w0[...,0]=0.92; w0[...,1]=0.03; w0[...,2]=0.05
cx,cy=int(nx*0.28), int(ny*0.55)
rr = ((np.arange(nx)[:,None]-cx)**2/(0.12*nx)**2 + (np.arange(ny)[None,:]-cy)**2/(0.14*ny)**2)
patch = np.exp(-rr)
w0[...,1]+=0.18*patch; w0[...,0]-=0.12*patch; w0[...,2]+=0.04*patch
w0=np.clip(w0,1e-6,None); w0/=w0.sum(axis=-1,keepdims=True)

wA  = apply_write(w0, barA, strength=0.72)
wAB = apply_write(wA, barB, strength=0.38)
wAB = apply_erase(wAB, barB, boundary_map, pulse_dir='B', erase_strength=0.58)
wAB3 = wAB.copy()
for _ in range(3):
    wAB3 = apply_write(wAB3, barB, strength=0.25)
    wAB3 = apply_erase(wAB3, barB, boundary_map, pulse_dir='B', erase_strength=0.40)

cube0  = render_cube_fast(w0, dom, theta_map, boundary_map, k, e, basis3, broaden_meV=0.0)
cubeA  = render_cube_fast(wA, dom, theta_map, boundary_map, k, e, basis3, broaden_meV=0.0)
cubeAB = render_cube_fast(wAB, dom, theta_map, boundary_map, k, e, basis3, broaden_meV=8.0)
cubeAB3= render_cube_fast(wAB3, dom, theta_map, boundary_map, k, e, basis3, broaden_meV=10.0)

Irat0=compute_Irat(cube0,e)
IratA=compute_Irat(cubeA,e)
IratAB=compute_Irat(cubeAB,e)
IratAB3=compute_Irat(cubeAB3,e)

# --- plotting helpers ---
def imshow(ax, img, title):
    im = ax.imshow(img.T, origin='lower', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return im

# Maps like Fig 7.2/7.3
fig,axs = plt.subplots(2,4, figsize=(12,6))
imshow(axs[0,0], Irat0, "Irat: pristine")
imshow(axs[0,1], IratA, "Irat: after pulse A")
imshow(axs[0,2], IratAB, "Irat: after pulse B (erase A)")
imshow(axs[0,3], IratAB3, "Irat: after +3 pulses B")

# difference maps (like 7.5/7.7)
dA = IratA - Irat0
dAB = IratAB - IratA
dAB3 = IratAB3 - IratAB
imshow(axs[1,0], dA, "ΔIrat (A - pristine)")
imshow(axs[1,1], dAB, "ΔIrat (B - A)")
imshow(axs[1,2], dAB3, "ΔIrat (+3B - B)")
imshow(axs[1,3], boundary_map, "domain-boundary proxy")
plt.tight_layout()
fig_path="synth_maps_v1.png"
plt.savefig(fig_path, dpi=200)
plt.close(fig)

# Show representative spectra from two ROIs (magenta/orange idea)
def roi_mean_cube(cube, x0,x1,y0,y1):
    return cube[x0:x1,y0:y1].mean(axis=(0,1))  # (Nk,Ne)

# choose ROIs: one inside A bar; one near interface/boundary
roi1 = roi_mean_cube(cubeA, 22,30, 18,26)  # "written metallic"
roi2 = roi_mean_cube(cubeAB, 40,48, 8,16)  # "partially erased / intermediate"

def plot_spectrum(spec, title, out):
    fig,ax=plt.subplots(1,1,figsize=(5,4))
    ax.imshow(spec.T, origin='lower', aspect='auto',
              extent=[k[0],k[-1],e[0],e[-1]])
    ax.set_xlabel("k (arb.)")
    ax.set_ylabel("E (eV)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)

plot_spectrum(roi1, "ROI1 spectrum (after A)", "synth_roi1_spectrum.png")
plot_spectrum(roi2, "ROI2 spectrum (after B)", "synth_roi2_spectrum.png")

# EDCs at k=0 and k=0.3
def edc(spec, k_target):
    idx = int(np.argmin(np.abs(k-k_target)))
    return spec[idx]

fig,ax=plt.subplots(1,1,figsize=(6,4))
for lbl, spec in [("pristine", roi_mean_cube(cube0,22,30,18,26)),
                  ("after A", roi1),
                  ("after B", roi_mean_cube(cubeAB,22,30,18,26)),
                  ("after +3B", roi_mean_cube(cubeAB3,22,30,18,26))]:
    ax.plot(e, edc(spec, 0.0), label=lbl)
ax.set_xlabel("E (eV)")
ax.set_ylabel("Intensity (arb.)")
ax.set_title("EDCs at k≈0 (ROI1 region over sequence)")
ax.legend()
plt.tight_layout()
edc_path="synth_edcs_roi1.png"
plt.savefig(edc_path, dpi=200)
plt.close(fig)

# Save a compact dataset (float16) quickly (uncompressed .npz)
outpath="synth_tas2_pulsing_v1.npz"
np.savez(outpath,
         k=k.astype(np.float32), e=e.astype(np.float32),
         Irat0=Irat0.astype(np.float32), IratA=IratA.astype(np.float32),
         IratAB=IratAB.astype(np.float32), IratAB3=IratAB3.astype(np.float32),
         w0=w0.astype(np.float16), wA=wA.astype(np.float16), wAB=wAB.astype(np.float16), wAB3=wAB3.astype(np.float16),
         dom=dom.astype(np.int16), theta_map=theta_map.astype(np.float16), boundary_map=boundary_map.astype(np.float16),
         barA=barA.astype(np.float16), barB=barB.astype(np.float16),
         # store only two cubes to keep file small
         cube0=cube0.astype(np.float16), cubeAB=cubeAB.astype(np.float16))

(fig_path, "synth_roi1_spectrum.png", "synth_roi2_spectrum.png", edc_path, outpath)
