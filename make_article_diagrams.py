# Draws the methodology (Figure 1) and protocol (Figure 4) diagrams of the revised manuscript.
# Run from the repository root: python make_article_diagrams.py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import os
OUT="figures_article/"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42})
def box(ax,x,y,w,h,title,lines,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=ec,lw=1.3))
    ax.text(x+w/2,y+h-0.17,title,ha="center",va="top",fontsize=9.5,weight="bold",color="#1a1a1a")
    ax.text(x+w/2,y+h-0.52,"\n".join(lines),ha="center",va="top",fontsize=7.9,color="#333333",linespacing=1.35)
def arr(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,lw=1.4,color="#555555"))
# ---- Figure 1: pipeline
fig,ax=plt.subplots(figsize=(12,3.5)); ax.set_xlim(0,12); ax.set_ylim(0,3.5); ax.axis("off")
B=[(0.1,"1. Data",["PandaSet: 39 seq. × 8 s","(Pandar64 + poses;","road labels in 30 seq.)","KITTI Road: 12 seq.","(HDL-64E + OXTS), test only"],"#eaf2fb","#0072B2"),
   (2.45,"2. Frame transform",["World → vehicle frame","p̃_v = T_vw p̃_w  (SE(3))","yaw from trajectory,","held when stopped"],"#eaf2fb","#0072B2"),
   (4.8,"3. Per-frame features",["Road geometry (two variants):","height band | semantic road","preview y_10..y_40, θ, grade,","curvature κ → R = 1/|κ|","Dynamics: v, dψ/dt"],"#fff4e5","#E69F00"),
   (7.15,"4. Samples",["Window W = 2 s (20 frames)","+ first differences of v, dψ/dt","Targets: mean dψ/dt, v over","H = 1, 2, 3 s (residuals)","v_max from last frame only"],"#fff4e5","#E69F00"),
   (9.5,"5. Predictors",["Persistence, constant trend","Ridge, MLP, LSTM, GRU","CNN-LSTM (λ = 0)","CNN-LSTM + physics term","(λ, μ sensitivity)"],"#e8f6f0","#009E73")]
for x,t,l,fc,ec in B: box(ax,x,1.45,2.2,2.0,t,l,fc,ec)
for i in range(4): arr(ax,(B[i][0]+2.22,2.45),(B[i+1][0]-0.02,2.45))
box(ax,1.2,0.05,9.6,1.05,"6. Evaluation",["5-fold sequence-wise cross-validation on PandaSet (each sequence tested once) · zero-shot KITTI with every fold model",
    "MSE, R², relative MSE vs. persistence · per-sequence Wilcoxon + Holm · bootstrap 95% CI · ablations · curve strata · geometry validation vs. future path"],"#f3f0f7","#7b5ea7")
arr(ax,(10.6,1.45),(10.6,1.12))
fig.savefig(OUT+"fig1_methodology.png",dpi=300,bbox_inches="tight"); fig.savefig(OUT+"fig1_methodology.pdf",bbox_inches="tight"); plt.close(fig)
# ---- Protocol: folds
fig,ax=plt.subplots(figsize=(11,3.4)); ax.set_xlim(0,11); ax.set_ylim(0,3.4); ax.axis("off")
C={"train":"#cfe2f3","val":"#ffe0b2","test":"#0072B2"}
for k in range(5):
    y=2.75-k*0.5
    ax.text(0.05,y+0.17,f"Fold {k+1}",fontsize=9,va="center")
    for j in range(5):
        x=0.9+j*1.25
        if j==k: c="test"
        else: c="train"
        ax.add_patch(Rectangle((x,y),1.2,0.34,fc=C[c],ec="white",lw=1.5))
    ax.text(0.9+k*1.25+0.6,y+0.17,"test (7–8 seq.)",ha="center",va="center",fontsize=7.5,color="white")
    ax.add_patch(Rectangle((7.25,y),0.9,0.34,fc=C["val"],ec="white"))
    ax.text(7.7,y+0.17,"val",ha="center",va="center",fontsize=7.5)
ax.text(0.9,3.2,"PandaSet sequences split into 5 disjoint groups (whole sequences)",fontsize=9)
ax.text(7.25,3.2,"15% of the non-test seq.\nfor early stopping",fontsize=8,va="bottom")
ax.add_patch(FancyBboxPatch((8.5,0.75),2.4,2.3,boxstyle="round,pad=0.02",fc="#e8f6f0",ec="#009E73"))
ax.text(9.7,2.85,"KITTI Road (12 seq.)",ha="center",va="top",fontsize=9,weight="bold")
ax.text(9.7,2.45,"never used for training,\nscaling or tuning;\nevaluated with each of\nthe 5 fold models\n(mean ± SD, pooled CI)",ha="center",va="top",fontsize=8,linespacing=1.35)
ax.text(0.9,0.25,"Scalers fitted on the training sequences of each fold only. Test predictions of the 5 folds are pooled: every sequence is predicted once by a model that never saw it.",fontsize=8,color="#444444")
fig.savefig(OUT+"fig_protocol.png",dpi=300,bbox_inches="tight"); fig.savefig(OUT+"fig_protocol.pdf",bbox_inches="tight"); plt.close(fig)
print("ok")
