import json, warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.95)
PAL={"Adelie":"#d95f02","Chinstrap":"#7570b3","Gentoo":"#1b9e77"}
R={}
def save(name):
    plt.savefig(f"figures/{name}.png",dpi=170,bbox_inches="tight"); plt.close()
raw=pd.read_csv("data/penguins.csv"); raw["sex"]=raw["sex"].str.title()
num=["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
R["shape"]=raw.shape; R["head"]=raw.head().to_string()
R["miss"]=raw.isna().sum().to_dict(); R["rows_missing"]=int(raw.isna().any(axis=1).sum())
R["rows_all4_missing"]=int(raw[num].isna().all(axis=1).sum())
R["rows_sex_only"]=int((raw[num].notna().all(axis=1)&raw.sex.isna()).sum())
R["dups"]=int(raw.duplicated().sum())
R["dtypes"]={k:str(v) for k,v in raw.dtypes.items()}
R["ranges"]={c:[float(raw[c].min()),float(raw[c].max())] for c in num}
R["raw_counts_species"]=raw.species.value_counts().to_dict()
# imputation comparison
imp=raw.copy()
for c in num: imp[c]=imp.groupby("species")[c].transform(lambda s:s.fillna(s.median()))
imp["sex"]=imp.groupby("species")["sex"].transform(lambda s:s.fillna(s.mode()[0]))
R["mean_mass_raw"]=float(raw.body_mass_g.mean()); R["mean_mass_imp"]=float(imp.body_mass_g.mean())
df=raw.dropna().reset_index(drop=True); R["clean_shape"]=df.shape

# Fig 3.2 missing
fig,ax=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={"width_ratios":[1,1.3]})
m=raw.isna().sum(); pc=m/len(raw)*100
ax[0].barh(m.index,m.values,color="#c0392b"); 
for i,(v,p) in enumerate(zip(m.values,pc.values)): ax[0].text(v+0.15,i,f"{v} ({p:.1f}%)",va="center",fontsize=8)
ax[0].set_xlim(0,14); ax[0].invert_yaxis(); ax[0].set_title("Null count per column"); ax[0].set_xlabel("Number of missing entries")
sns.heatmap(raw.isna().T,cbar=False,cmap=["#ecf0f1","#c0392b"],ax=ax[1],xticklabels=False); ax[1].set_title("Row-wise null pattern (red = missing)"); ax[1].set_xlabel("Row index (0-343)")
plt.tight_layout(); save("fig3_2_missing")
# Fig 3.1 process cycle
fig,ax=plt.subplots(figsize=(7,4.4)); ax.axis("off"); ax.set_xlim(-1.6,1.6); ax.set_ylim(-1.35,1.35)
steps=["Define\nproblem","Acquire\ndata","Clean &\nprepare","Explore &\nvisualize","Model &\nvalidate","Report\nfindings"]
cols=["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51","#6d597a"]
for i,(s,c) in enumerate(zip(steps,cols)):
    a=np.pi/2-i*2*np.pi/6; x,y=1.15*np.cos(a)*1.15,1.0*np.sin(a)
    ax.text(x,y,s,ha="center",va="center",fontsize=9,color="white" if c not in("#e9c46a",) else "black",fontweight="bold",bbox=dict(boxstyle="round,pad=0.5",fc=c,ec="none"))
    a2=np.pi/2-(i+1)*2*np.pi/6; x2,y2=1.15*np.cos(a2)*1.15,1.0*np.sin(a2)
    ax.annotate("",xy=(x2*0.8+x*0.2,y2*0.8+y*0.2),xytext=(x*0.8+x2*0.2,y*0.8+y2*0.2),arrowprops=dict(arrowstyle="->",color="gray",lw=1.4,connectionstyle="arc3,rad=-0.25"))
ax.text(0,0,"EDA sits at the\nheart of the loop;\nfindings feed back\ninto cleaning",ha="center",va="center",fontsize=9,style="italic")
save("fig3_1_process")
# transformations
body=df.body_mass_g
fig,ax=plt.subplots(1,4,figsize=(12,2.9))
tr=[("Raw body mass (g)",body,"#264653"),("log(body mass)",np.log(body),"#2a9d8f"),("sqrt(body mass)",np.sqrt(body),"#e9a23b"),("Z-score",(body-body.mean())/body.std(),"#e76f51")]
for a,(t,v,c) in zip(ax,tr):
    sns.histplot(v,bins=15,kde=True,color=c,ax=a); a.set_title(t,fontsize=10); a.set_xlabel("")
R["skew_body"]=float(stats.skew(body)); R["skew_log"]=float(stats.skew(np.log(body))); R["skew_sqrt"]=float(stats.skew(np.sqrt(body))); R["skew_z"]=float(stats.skew((body-body.mean())/body.std()))
plt.tight_layout(); save("fig3_3_transform")
R["qcut"]=pd.qcut(body,3,labels=["Light","Medium","Heavy"]).value_counts().to_dict()
R["qcut_edges"]=[float(x) for x in pd.qcut(body,3).cat.categories.left]+[float(pd.qcut(body,3).cat.categories.right[-1])]
df["bill_ratio"]=df.bill_length_mm/df.bill_depth_mm
R["ratio_by_sp"]=df.groupby("species").bill_ratio.mean().round(2).to_dict()
# encoding
enc=df.copy(); R["enc_species"]={"Adelie":0,"Chinstrap":1,"Gentoo":2}
R["sex_counts"]=df.sex.value_counts().to_dict()
R["slice_heavy"]=int(((df.species=="Gentoo")&(df.body_mass_g>5500)).sum())
R["slice_heavy_max"]=float(df[df.species=="Gentoo"].body_mass_g.max())
R["q_female_dream"]=int(df.query("sex=='Female' and island=='Dream'").shape[0])
R["slice_small"]=int((df.body_mass_g<3000).sum())
R["slice_small_sp"]=df[df.body_mass_g<3000].species.value_counts().to_dict()
# grouping
g=df.groupby("species")[num].mean().round(2); R["group_mean"]=g.to_dict("index")
R["group_cnt"]=df.species.value_counts().to_dict()
R["group_sd"]=df.groupby("species")[num].std().round(2).to_dict("index")
R["xtab"]=pd.crosstab(df.species,df.island,margins=True).to_dict()
R["pivot_mass"]=df.pivot_table(values="body_mass_g",index="species",columns="island",aggfunc="mean").round(1).to_dict()
R["sex_species_mass"]=df.groupby(["species","sex"]).body_mass_g.mean().round(1).unstack().to_dict("index")
R["species_counts"]=df.species.value_counts().to_dict()

# ---------- Unit 2
# 4.1
fig,ax=plt.subplots(1,3,figsize=(13,3.9))
bins=pd.cut(df.flipper_length_mm,bins=range(170,236,5)); lm=df.groupby(bins).body_mass_g.mean().dropna()
mid=[iv.mid for iv in lm.index]
ax[0].plot(mid,lm.values,"o-",color="#264653",label="Mean mass per 5 mm bin"); ax[0].plot(mid,pd.Series(lm.values).rolling(3,center=True).mean(),"--",color="#e76f51",label="3-bin moving average")
ax[0].set_xlabel("Flipper length (mm, bin centre)"); ax[0].set_ylabel("Body mass (g)"); ax[0].set_title("(a) Line plot"); ax[0].legend(fontsize=8)
for sp,gg in df.groupby("species"): ax[1].scatter(gg.bill_length_mm,gg.bill_depth_mm,s=(gg.body_mass_g-2500)/40,c=PAL[sp],alpha=.65,label=sp,edgecolor="white",lw=.4)
ax[1].set_xlabel("Bill length (mm)"); ax[1].set_ylabel("Bill depth (mm)"); ax[1].set_title("(b) Scatter plot (marker size = body mass)"); ax[1].legend(title="Species",fontsize=8,markerscale=.6)
st=df.groupby("island").flipper_length_mm.agg(["mean","std","count"]); st["ci"]=1.96*st["std"]/np.sqrt(st["count"])
x=np.arange(3); ax[2].errorbar(x-.1,st["mean"],yerr=st["std"],fmt="s",color="#264653",capsize=6,label="Mean ± 1 SD")
ax[2].errorbar(x+.1,st["mean"],yerr=st["ci"],fmt="o",color="#e76f51",capsize=3,label="Mean ± 95% CI")
ax[2].set_xticks(x); ax[2].set_xticklabels(st.index); ax[2].set_ylabel("Flipper length (mm)"); ax[2].set_title("(c) Error plot by island"); ax[2].legend(fontsize=8)
plt.tight_layout(); save("fig4_1_basic")
R["island_flip"]=st.round(2).to_dict("index")
# 4.2
fig,ax=plt.subplots(1,3,figsize=(13,3.9))
hb=ax[0].hexbin(df.bill_length_mm,df.body_mass_g,gridsize=18,cmap="YlGnBu",mincnt=1); plt.colorbar(hb,ax=ax[0],label="Count")
ax[0].set_xlabel("Bill length (mm)"); ax[0].set_ylabel("Body mass (g)"); ax[0].set_title("(a) Hexbin density")
sns.kdeplot(data=df,x="bill_length_mm",y="bill_depth_mm",fill=True,cmap="mako_r",levels=10,thresh=.03,ax=ax[1]); sns.kdeplot(data=df,x="bill_length_mm",y="bill_depth_mm",color="white",levels=10,linewidths=.6,ax=ax[1])
ax[1].set_xlabel("Bill length (mm)"); ax[1].set_ylabel("Bill depth (mm)"); ax[1].set_title("(b) Filled contour of 2-D KDE")
for b,ls,c in [(5,"-","#264653"),(15,"--","#e76f51"),(40,":","#2a9d8f")]: ax[2].hist(df.flipper_length_mm,bins=b,histtype="step",ls=ls,lw=1.8,color=c,label=f"{b} bins")
ax[2].set_xlabel("Flipper length (mm)"); ax[2].set_ylabel("Frequency"); ax[2].set_title("(c) Histogram bin-width sensitivity"); ax[2].legend()
plt.tight_layout(); save("fig4_2_density")
# 4.3
fig,ax=plt.subplots(2,2,figsize=(11,8))
a=ax[0,0]
for sp,gg in df.groupby("species"): a.scatter(gg.flipper_length_mm,gg.bill_length_mm,c=PAL[sp],label=sp,alpha=.7,s=22)
a.legend(title="Species",loc="lower right",frameon=True,shadow=True,fancybox=True); a.set_title("(a) Legend: title, location, shadow"); a.set_xlabel("Flipper length (mm)"); a.set_ylabel("Bill length (mm)")
a=ax[0,1]; sc=a.scatter(df.bill_depth_mm,df.body_mass_g,c=df.flipper_length_mm,cmap="viridis",s=25); plt.colorbar(sc,ax=a,label="Flipper length (mm)"); a.set_title("(b) Continuous colour map: viridis"); a.set_xlabel("Bill depth (mm)"); a.set_ylabel("Body mass (g)")
a=ax[1,0]; a.hist([df[df.species==s].body_mass_g for s in PAL],bins=15,stacked=True,color=list(PAL.values()),label=list(PAL),edgecolor="white"); a.legend(title="Species"); a.set_title("(c) Stacked histogram, custom colours"); a.set_xlabel("Body mass (g)"); a.set_ylabel("Frequency")
a=ax[1,1]; a.scatter(df.flipper_length_mm,df.body_mass_g,c=[PAL[s] for s in df.species],s=18,alpha=.6)
hv=df.loc[df.body_mass_g.idxmin()]; a.annotate(f"Lightest bird: {int(hv.body_mass_g)} g\n({hv.species})",xy=(hv.flipper_length_mm,hv.body_mass_g),xytext=(200,2800),arrowprops=dict(arrowstyle="->",color="k"),fontsize=8)
a.axhline(df.body_mass_g.mean(),ls="--",color="gray"); a.text(172,df.body_mass_g.mean()+80,f"Mean = {df.body_mass_g.mean():.0f} g",fontsize=8,color="gray")
a.axvspan(df.flipper_length_mm.quantile(.9),232,alpha=.12,color="#1b9e77"); a.text(df.flipper_length_mm.quantile(.9)+.5,6000,"Top 10%\nflippers",fontsize=8,color="#1b9e77")
a.set_title("(d) Text, annotation, reference line, shaded span"); a.set_xlabel("Flipper length (mm)"); a.set_ylabel("Body mass (g)")
plt.tight_layout(); save("fig4_3_style")
R["lightest"]=[int(hv.body_mass_g),hv.species]; R["flip_q90"]=float(df.flipper_length_mm.quantile(.9))
# 4.4 subplots 2x2 (shared axes)
fig,ax=plt.subplots(2,2,figsize=(10,7),sharey=False)
for a,c in zip(ax.ravel(),num):
    for sp in PAL: sns.kdeplot(df[df.species==sp][c],ax=a,color=PAL[sp],fill=True,alpha=.3,label=sp)
    a.set_title(c.replace("_"," "),fontsize=10); a.set_xlabel("")
ax[0,0].legend(fontsize=8); fig.suptitle("Species-wise density of each measurement (2 x 2 subplot grid)",y=1.0,fontsize=11); plt.tight_layout(); save("fig4_4_subplots")
# 4.5 3D
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
fig=plt.figure(figsize=(12,5.2)); a=fig.add_subplot(121,projection="3d")
for sp,gg in df.groupby("species"): a.scatter(gg.bill_length_mm,gg.flipper_length_mm,gg.body_mass_g,c=PAL[sp],label=sp,s=14,alpha=.85)
a.set_xlabel("Bill length (mm)"); a.set_ylabel("Flipper length (mm)"); a.set_zlabel("Body mass (g)"); a.legend(title="Species",fontsize=8,loc="upper left"); a.set_title("(a) 3-D scatter"); a.view_init(18,-125)
a=fig.add_subplot(122,projection="3d")
kd=gaussian_kde(np.vstack([df.flipper_length_mm,df.body_mass_g])); gx=np.linspace(168,236,60); gy=np.linspace(2300,6800,60); X,Y=np.meshgrid(gx,gy)
Zd=kd(np.vstack([X.ravel(),Y.ravel()])).reshape(X.shape)*1e4
a.plot_surface(X,Y,Zd,cmap="magma",edgecolor="none",alpha=.95); a.set_xlabel("Flipper length (mm)"); a.set_ylabel("Body mass (g)"); a.set_zlabel("Density (x1e-4)"); a.set_title("(b) Surface of the 2-D kernel density"); a.view_init(30,-60)
plt.tight_layout(); save("fig4_5_3d")
# 4.6 seaborn
fig,ax=plt.subplots(2,2,figsize=(11,8))
sns.scatterplot(data=df,x="flipper_length_mm",y="bill_length_mm",hue="species",style="sex",palette=PAL,ax=ax[0,0]); ax[0,0].set_title("(a) scatterplot: hue = species, style = sex")
sns.ecdfplot(data=df,x="body_mass_g",hue="species",palette=PAL,ax=ax[0,1]); ax[0,1].set_title("(b) ecdfplot of body mass")
sns.regplot(data=df,x="bill_length_mm",y="body_mass_g",scatter_kws=dict(s=14,alpha=.5,color="#264653"),line_kws=dict(color="#e76f51"),ax=ax[1,0]); ax[1,0].set_title("(c) regplot with 95% band")
sns.stripplot(data=df,x="island",y="body_mass_g",hue="species",palette=PAL,dodge=True,size=3.5,alpha=.7,ax=ax[1,1]); ax[1,1].set_title("(d) stripplot by island and species")
plt.tight_layout(); save("fig4_6_seaborn")
# plotly + bokeh html
import plotly.express as px
fig=px.scatter(df,x="flipper_length_mm",y="body_mass_g",color="species",symbol="sex",hover_data=["island","bill_length_mm","bill_depth_mm"],color_discrete_map=PAL,marginal_x="histogram",marginal_y="box",title="Palmer Penguins: flipper length vs body mass")
fig.write_html("outputs/interactive/plotly_scatter.html",include_plotlyjs="cdn")
fig3=px.scatter_3d(df,x="bill_length_mm",y="bill_depth_mm",z="flipper_length_mm",color="species",color_discrete_map=PAL,size="body_mass_g",size_max=9); fig3.write_html("outputs/interactive/plotly_3d.html",include_plotlyjs="cdn")
from bokeh.plotting import figure, output_file, save as bsave
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.transform import factor_cmap
src=ColumnDataSource(df); output_file("outputs/interactive/bokeh_scatter.html")
p=figure(width=760,height=480,title="Bill length vs bill depth (Bokeh)",tools="pan,wheel_zoom,box_zoom,reset,save")
p.scatter("bill_length_mm","bill_depth_mm",source=src,size=7,alpha=.8,legend_field="species",color=factor_cmap("species",list(PAL.values()),list(PAL)))
p.add_tools(HoverTool(tooltips=[("Species","@species"),("Island","@island"),("Mass (g)","@body_mass_g")])); p.xaxis.axis_label="Bill length (mm)"; p.yaxis.axis_label="Bill depth (mm)"; bsave(p)

# ---------- Unit 3
rows=[]
for c in num:
    s=df[c]; q1,q3=s.quantile([.25,.75]); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr
    z=np.abs(stats.zscore(s)); sw=stats.shapiro(s)
    rows.append(dict(var=c,mean=s.mean(),median=s.median(),std=s.std(),min=s.min(),max=s.max(),q1=q1,q3=q3,iqr=iqr,skew=stats.skew(s),kurt=stats.kurtosis(s),lo=lo,hi=hi,out_iqr=int(((s<lo)|(s>hi)).sum()),out_z=int((z>3).sum()),shp=sw.pvalue,shW=sw.statistic,cv=s.std()/s.mean()*100))
R["uni"]=rows
fig,ax=plt.subplots(2,2,figsize=(10,7)); cl=["#264653","#2a9d8f","#e9a23b","#e76f51"]
for a,c,k in zip(ax.ravel(),num,cl):
    sns.histplot(df[c],bins=20,kde=True,color=k,ax=a); a.axvline(df[c].mean(),color="red",ls="--",label=f"Mean {df[c].mean():.1f}"); a.axvline(df[c].median(),color="black",ls=":",label=f"Median {df[c].median():.1f}"); a.legend(fontsize=8); a.set_title(f"{c}  (skew = {stats.skew(df[c]):.2f})",fontsize=10)
plt.tight_layout(); save("fig5_1_hist")
fig,ax=plt.subplots(1,4,figsize=(13,3.6))
for a,c,k in zip(ax,num,cl): sns.boxplot(y=df[c],color=k,ax=a,width=.4); a.set_title(c.replace("_"," "),fontsize=10)
plt.tight_layout(); save("fig5_2_box")
fig,ax=plt.subplots(1,4,figsize=(13,3.4))
for a,c,k in zip(ax,num,cl): stats.probplot(df[c],dist="norm",plot=a); a.get_lines()[0].set_color(k); a.get_lines()[0].set_markersize(3); a.set_title("Q-Q: "+c.replace("_"," "),fontsize=10); a.set_xlabel("Theoretical quantiles"); a.set_ylabel("Ordered values")
plt.tight_layout(); save("fig5_3_qq")
cat={}
fig,ax=plt.subplots(2,3,figsize=(12,6.8)); cc=["#264653","#2a9d8f","#e9a23b"]
for j,c in enumerate(["species","island","sex"]):
    vc=df[c].value_counts(); cat[c]={k:[int(v),round(v/len(df)*100,1)] for k,v in vc.items()}
    ax[0,j].bar(vc.index,vc.values,color=cc[:len(vc)]); ax[0,j].set_title(f"Bar chart: {c}")
    for i,v in enumerate(vc.values): ax[0,j].text(i,v+2,str(v),ha="center",fontsize=9)
    ax[1,j].pie(vc.values,labels=vc.index,autopct="%1.1f%%",colors=cc[:len(vc)],startangle=90,wedgeprops=dict(edgecolor="white")); ax[1,j].set_title(f"Pie chart: {c}")
plt.tight_layout(); save("fig5_4_cat"); R["cat"]=cat

# ---------- Unit 4
cm=df[num].corr(); sp_=df[num].corr(method="spearman")
pairs=[]
import itertools
for a_,b_ in itertools.combinations(num,2):
    r,p=stats.pearsonr(df[a_],df[b_]); rho,p2=stats.spearmanr(df[a_],df[b_]); pairs.append([a_,b_,r,rho,p,p2])
R["pairs"]=pairs
fig,ax=plt.subplots(1,3,figsize=(13,3.9))
sns.regplot(data=df,x="flipper_length_mm",y="body_mass_g",scatter_kws=dict(s=14,alpha=.5,color="#264653"),line_kws=dict(color="#e76f51"),ax=ax[0]); ax[0].set_title("(a) Flipper vs body mass")
sns.regplot(data=df,x="bill_length_mm",y="bill_depth_mm",scatter_kws=dict(s=14,alpha=.5,color="gray"),line_kws=dict(color="black"),ax=ax[1]); ax[1].set_title("(b) Bill length vs depth (pooled)")
for sp,gg in df.groupby("species"): sns.regplot(data=gg,x="bill_length_mm",y="bill_depth_mm",scatter_kws=dict(s=14,alpha=.55),color=PAL[sp],label=sp,ax=ax[2])
ax[2].legend(title="Species",fontsize=8); ax[2].set_title("(c) Bill length vs depth, per species")
plt.tight_layout(); save("fig6_1_scatter")
R["within_bill"]={sp:float(gg.bill_length_mm.corr(gg.bill_depth_mm)) for sp,gg in df.groupby("species")}
R["within_flip_mass"]={sp:float(gg.flipper_length_mm.corr(gg.body_mass_g)) for sp,gg in df.groupby("species")}
R["within_corr"]={sp:gg[num].corr().round(2).values.tolist() for sp,gg in df.groupby("species")}
# cat vs cat
ct=pd.crosstab(df.species,df.island); chi,p,dof,_=stats.chi2_contingency(ct); V=np.sqrt(chi/(len(df)*(min(ct.shape)-1)))
R["chi_si"]=[chi,p,dof,V]
ct2=pd.crosstab(df.species,df.sex); chi2,p2,d2,_=stats.chi2_contingency(ct2); R["chi_ss"]=[chi2,p2,d2]
ct3=pd.crosstab(df.island,df.sex); R["chi_is"]=list(stats.chi2_contingency(ct3)[:3]); R["xtab_sex"]=ct2.to_dict()
fig,ax=plt.subplots(1,3,figsize=(13,3.8))
ct.plot(kind="bar",stacked=True,ax=ax[0],color=cc,rot=0); ax[0].set_title("(a) Stacked: species by island")
ct.plot(kind="bar",ax=ax[1],color=cc,rot=0); ax[1].set_title("(b) Grouped: species by island")
(ct2.div(ct2.sum(1),axis=0)*100).plot(kind="bar",ax=ax[2],color=["#c0392b","#2980b9"],rot=0); ax[2].set_title("(c) Percentage of each sex within species"); ax[2].set_ylabel("Percent")
for a in ax: a.set_xlabel("Species")
plt.tight_layout(); save("fig6_2_catcat")
# num vs cat
fig,ax=plt.subplots(2,4,figsize=(15,7))
for j,c in enumerate(num):
    sns.boxplot(data=df,x="species",y=c,palette=PAL,ax=ax[0,j],width=.55); ax[0,j].set_xlabel("")
    sns.violinplot(data=df,x="species",y=c,hue="sex",split=True,inner="quartile",palette={"Female":"#e07a9a","Male":"#4c78a8"},ax=ax[1,j]); ax[1,j].set_xlabel("")
    if j: ax[1,j].get_legend().remove()
plt.tight_layout(); save("fig6_3_boxviolin")
an=[]
for c in num:
    gs=[df[df.species==s][c] for s in PAL]; F,p=stats.f_oneway(*gs); H,pk=stats.kruskal(*gs)
    ssb=sum(len(x)*(x.mean()-df[c].mean())**2 for x in gs); sst=((df[c]-df[c].mean())**2).sum(); an.append([c,F,p,ssb/sst,H,pk])
R["anova"]=an
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tk=pairwise_tukeyhsd(df.body_mass_g,df.species); R["tukey_mass"]=str(tk.summary())
tk2=pairwise_tukeyhsd(df.bill_length_mm,df.species); R["tukey_bill"]=str(tk2.summary())
tk3=pairwise_tukeyhsd(df.flipper_length_mm,df.species); R["tukey_flip"]=str(tk3.summary())
mf=df[df.sex=="Male"].body_mass_g; ff=df[df.sex=="Female"].body_mass_g; t,p=stats.ttest_ind(mf,ff,equal_var=False)
R["welch"]=[mf.mean(),ff.mean(),t,p]
R["welch_sp"]={sp:[*stats.ttest_ind(gg[gg.sex=="Male"].body_mass_g,gg[gg.sex=="Female"].body_mass_g,equal_var=False)] for sp,gg in df.groupby("species")}
R["dimorph_pct"]={sp:float((gg[gg.sex=="Male"].body_mass_g.mean()/gg[gg.sex=="Female"].body_mass_g.mean()-1)*100) for sp,gg in df.groupby("species")}

# ---------- Unit 5
sns.pairplot(df,vars=num,hue="species",palette=PAL,diag_kind="kde",plot_kws=dict(s=14,alpha=.65),height=2.0); save("fig7_1_pair")
fig,ax=plt.subplots(2,2,figsize=(10,8.2)); lab=["bill len","bill dep","flip len","mass"]
sets=[("All penguins",df)]+[(s,df[df.species==s]) for s in PAL]
for a,(t,d) in zip(ax.ravel(),sets):
    sns.heatmap(d[num].corr(),annot=True,fmt=".2f",cmap="RdBu_r",vmin=-1,vmax=1,ax=a,xticklabels=lab,yticklabels=lab,cbar=False,square=True); a.set_title(t)
plt.tight_layout(); save("fig7_2_heat")
# grouped
fig,ax=plt.subplots(1,3,figsize=(14,4.2))
sns.pointplot(data=df,x="species",y="body_mass_g",hue="sex",dodge=.3,errorbar="sd",capsize=.1,palette={"Female":"#e07a9a","Male":"#4c78a8"},ax=ax[0]); ax[0].set_title("(a) Mean ± SD of mass by species and sex")
sns.barplot(data=df,x="island",y="body_mass_g",hue="species",palette=PAL,errorbar="sd",capsize=.1,ax=ax[1]); ax[1].set_title("(b) Mean mass by island and species (± SD)")
pv=df.pivot_table(values="bill_depth_mm",index="species",columns="sex",aggfunc="mean"); sns.heatmap(pv,annot=True,fmt=".1f",cmap="YlOrBr",ax=ax[2],cbar_kws=dict(label="mm")); ax[2].set_title("(c) Mean bill depth: species x sex")
plt.tight_layout(); save("fig7_3_grouped")
R["pivot_depth"]=pv.round(2).to_dict("index")
g=sns.FacetGrid(df,col="island",row="sex",hue="species",palette=PAL,height=2.5,aspect=1.2,margin_titles=True); g.map_dataframe(sns.scatterplot,x="bill_length_mm",y="body_mass_g",s=16,alpha=.8); g.add_legend(title="Species"); g.set_axis_labels("Bill length (mm)","Body mass (g)"); save("fig7_4_facet")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
Xs=StandardScaler().fit_transform(df[num]); pca=PCA().fit(Xs); Z=pca.transform(Xs)
R["pca_var"]=(pca.explained_variance_ratio_*100).tolist(); R["pca_load"]=pca.components_.tolist(); R["pca_eig"]=pca.explained_variance_.tolist()
ts=TSNE(2,perplexity=30,init="pca",random_state=42).fit_transform(Xs)
fig,ax=plt.subplots(2,2,figsize=(11,8.5))
ax[0,0].bar(range(1,5),pca.explained_variance_ratio_*100,color="#264653"); ax[0,0].plot(range(1,5),np.cumsum(pca.explained_variance_ratio_)*100,"o-",color="#e76f51"); ax[0,0].set_title("(a) Scree plot with cumulative variance"); ax[0,0].set_xlabel("Component"); ax[0,0].set_ylabel("Variance explained (%)"); ax[0,0].set_xticks(range(1,5))
sns.heatmap(pca.components_.T,annot=True,fmt=".2f",cmap="PuOr",center=0,yticklabels=lab,xticklabels=["PC1","PC2","PC3","PC4"],ax=ax[0,1],cbar=False); ax[0,1].set_title("(b) Component loadings")
for sp in PAL:
    m_=(df.species==sp).values; ax[1,0].scatter(Z[m_,0],Z[m_,1],c=PAL[sp],s=16,alpha=.75,label=sp); ax[1,1].scatter(ts[m_,0],ts[m_,1],c=PAL[sp],s=16,alpha=.75,label=sp)
ax[1,0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"); ax[1,0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"); ax[1,0].set_title("(c) PCA projection"); ax[1,0].legend(title="Species",fontsize=8)
ax[1,1].set_xlabel("t-SNE 1"); ax[1,1].set_ylabel("t-SNE 2"); ax[1,1].set_title("(d) t-SNE projection (perplexity = 30)")
plt.tight_layout(); save("fig7_5_pca")
# models
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
X=df[num].copy(); X["sex"]=df.sex.map({"Female":0,"Male":1}); y=df.species
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
R["split"]=[len(Xtr),len(Xte)]
models={"Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),"k-Nearest Neighbours":make_pipeline(StandardScaler(),KNeighborsClassifier(5)),"SVM (RBF)":make_pipeline(StandardScaler(),SVC()),"Decision Tree":DecisionTreeClassifier(random_state=42,max_depth=4),"Random Forest":RandomForestClassifier(300,random_state=42)}
cv=StratifiedKFold(5,shuffle=True,random_state=42); res={}
for n,m_ in models.items():
    sc=cross_val_score(m_,Xtr,ytr,cv=cv); m_.fit(Xtr,ytr); res[n]=[sc.mean()*100,sc.std()*100,accuracy_score(yte,m_.predict(Xte))*100]
R["models"]=res
best=models["Logistic Regression"]; cmx=confusion_matrix(yte,best.predict(Xte),labels=list(PAL)); R["cm_lr"]=cmx.tolist()
rf=models["Random Forest"]; R["rf_cm"]=confusion_matrix(yte,rf.predict(Xte),labels=list(PAL)).tolist()
R["rf_imp"]=dict(zip(X.columns,rf.feature_importances_.round(3).tolist()))
R["lr_report"]=classification_report(yte,best.predict(Xte))
fig,ax=plt.subplots(1,3,figsize=(14,4))
sns.heatmap(cmx,annot=True,fmt="d",cmap="Blues",xticklabels=list(PAL),yticklabels=list(PAL),ax=ax[0],cbar=False); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual"); ax[0].set_title("(a) Confusion matrix: logistic regression")
im=pd.Series(rf.feature_importances_,index=X.columns).sort_values(); ax[1].barh(im.index,im.values,color="#2a9d8f"); ax[1].set_title("(b) Random forest feature importance")
ax[2].bar(range(5),[v[0] for v in res.values()],yerr=[v[1] for v in res.values()],color="#264653",capsize=4); ax[2].set_xticks(range(5)); ax[2].set_xticklabels([k.replace(" ","\n") for k in res],fontsize=7); ax[2].set_ylim(90,101); ax[2].set_ylabel("5-fold CV accuracy (%)"); ax[2].set_title("(c) Model comparison")
plt.tight_layout(); save("fig7_6_models")
df.to_csv("data/penguins_clean.csv",index=False)
json.dump(R,open("results_summary.json","w"),indent=1,default=lambda o: o.item() if hasattr(o,"item") else str(o))
print("done")
