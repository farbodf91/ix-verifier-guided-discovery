"""Figures for scientific interpretation; no computation is hidden in notebook styling."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import find_peaks
from . import workflow as w

COLORS = {"data": "#242424", "baseline": "#b36117", "raw": "#7b6bb0",
          "refined": "#007e87", "alternative": "#c35755"}


def style():
    plt.rcParams.update({"figure.dpi": 125, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 11, "axes.titlesize": 12,
                         "legend.fontsize": 9, "axes.titlepad": 12,
                         "savefig.bbox": "tight"})


def intro():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    x = np.linspace(-2, 2, 17)
    ax[0].scatter(x, x*x+.1*np.random.default_rng(2).normal(size=len(x)), s=25, color=COLORS["refined"], label="observations")
    xx = np.linspace(-2.1, 2.1, 200)
    ax[0].plot(xx, xx**2, color=COLORS["data"], label=r"$y=x^2$")
    ax[0].set(xlabel="$x$", ylabel="$y$", title="Functional: a relation between variables")
    ax[0].legend(frameon=False)
    t = np.linspace(0, 4*np.pi, 240)
    ax[1].plot(t, np.cos(t), color=COLORS["refined"], label="$x(t)$")
    ax[1].plot(t, -np.sin(t), "--", color=COLORS["raw"], label="$v(t)$")
    ax[1].set(xlabel="Time", ylabel="State", title=r"Dynamical: $\dot{x}=v,\quad\dot{v}=-x$")
    ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2)
    return fig


def reference(case):
    tr = case["train"][0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    for j in range(2):
        ax[0].plot(tr["t"], tr["x"][:, j], label=f"$x_{j}$")
    ax[0].set(xlabel="Time", ylabel="State", title=rf"Van der Pol, $\mu={case['mu']:g}$")
    ax[0].legend(frameon=False)
    ax[1].plot(*tr["x"].T, color=COLORS["data"])
    ax[1].scatter(*tr["x0"], color=COLORS["refined"], s=35, label="initial state")
    ax[1].set(xlabel="$x_0$", ylabel="$x_1$", title="Phase portrait: position in state space")
    ax[1].legend(frameon=False)
    return fig


def split(case):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for i, tr in enumerate(case["train"]):
        ax[0].plot(*tr["x"].T, alpha=.35, color=COLORS["refined"], lw=1)
        ax[0].scatter(*tr["x0"], s=20, color=COLORS["refined"])
    for tr in case["test"]:
        ax[0].scatter(*tr["x0"], marker="x", s=30, color=COLORS["baseline"])
    ax[0].plot([], [], "o", color=COLORS["refined"], label="12 discovery initial states")
    ax[0].plot([], [], "x", color=COLORS["baseline"], label="8 held-out initial states")
    ax[0].set(xlabel="$x_0$", ylabel="$x_1$", title="Different starting states constrain one law")
    ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.24))
    tr=case["train"][8]
    ax[1].plot(tr["t"], tr["x"][:, 0], color=COLORS["data"])
    ax[1].axvspan(0,10,color=COLORS["raw"],alpha=.15)
    ax[1].axvspan(10,20,color=COLORS["refined"],alpha=.12)
    ax[1].axvline(10, color="0.6", lw=.8)
    ax[1].text(5, -.29,"early window",ha="center",va="top",transform=ax[1].get_xaxis_transform(), color=COLORS["raw"])
    ax[1].text(15,-.29,"late window",ha="center",va="top",transform=ax[1].get_xaxis_transform(), color=COLORS["refined"])
    ax[1].set(xlabel="Time", ylabel="$x_0$", title="12 trajectories give 24 discovery windows")
    return fig


def baseline(case, models):
    source=case["ranking_windows"][models["baseline_window"]]
    test=case["test"][models["test_index"]]
    fig, ax=plt.subplots(1,3,figsize=(12,3.2),layout="constrained")
    for a, tr, title in [(ax[0],source,"Supplied discovery window"),(ax[1],test,"Same equation, unseen initial state")]:
        p=w.rollout(models["baseline_raw"],tr["t"],tr["x0"])
        a.plot(tr["t"],tr["x"][:,0],color=COLORS["data"],label="data")
        if p is not None:a.plot(tr["t"],p[:,0],"--",color=COLORS["baseline"],label="single-trajectory model")
        else:a.text(.5,.1,"Rollout failed / timed out",ha="center",transform=a.transAxes)
        a.set(xlabel="Time",ylabel="$x_0$",title=title)
    p=w.rollout(models["baseline_raw"],test["t"],test["x0"])
    ax[2].plot(*test["x"].T,color=COLORS["data"])
    if p is not None:ax[2].plot(*p.T,"--",color=COLORS["baseline"])
    ax[2].set(xlabel="$x_0$",ylabel="$x_1$",title="Held-out phase portrait")
    fig.legend(*ax[0].get_legend_handles_labels(),loc="outside lower center",ncol=2,frameon=False)
    return fig


def decoding_settings(test, results):
    fig, axes = plt.subplots(1, len(results), figsize=(12, 3.6), layout="constrained")
    for ax, row in zip(np.atleast_1d(axes), results.itertuples()):
        equation = row.equation if isinstance(row.equation, str) else ""
        prediction = w.rollout(equation, test["t"], test["x0"]) if equation else None
        ax.plot(test["t"], test["x"][:, 0], color=COLORS["data"], label="observations")
        if prediction is not None:
            ax.plot(test["t"], prediction[:, 0], "--", color=COLORS["baseline"], label="selected equation")
        else:
            ax.text(.5, .5, "No complete rollout", ha="center", transform=ax.transAxes)
        ax.set(xlabel="Time", ylabel="$x_0$",
               title=f"Budget {row.beam_size}; temperature {row.temperature:g}")
    fig.suptitle("Same unseen initial state; equation selected using the input window only", fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    return fig


def pooling_comparison(case, models, pooled_equation):
    """Compare raw equations before verification or coefficient refinement."""
    test = case["test"][models["test_index"]]
    baseline = w.rollout(models["baseline_raw"], test["t"], test["x0"])
    pooled = w.rollout(pooled_equation, test["t"], test["x0"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    curves = [("Observations", test["x"], COLORS["data"], "-"),
              ("Single-trajectory baseline", baseline, COLORS["baseline"], "--"),
              ("Pooling + trajectory ranking", pooled, COLORS["raw"], "-.")]
    for j, ax in enumerate(axes):
        for label, values, color, linestyle in curves:
            if values is not None:
                ax.plot(test["t"], values[:, j], linestyle, color=color, label=label, lw=1.6)
        ax.set(xlabel="Time", ylabel=rf"$x_{j}$",
               title="Position-like coordinate" if j == 0 else "Velocity-like coordinate")
    fig.suptitle("Same unseen initial state: before and after pooling", fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    return fig


def derivatives(case, equation):
    tr=case["ranking_windows"][16]
    observed=np.gradient(tr["x"],tr["t"],axis=0,edge_order=2)
    predicted=w.vector_field(equation,tr["x"])
    fig,ax=plt.subplots(1,2,figsize=(10,3.4),layout="constrained")
    for j,a in enumerate(ax):
        a.plot(tr["t"],observed[:,j],color=COLORS["data"],label="slope estimated from data")
        a.plot(tr["t"],predicted[:,j],"--",color=COLORS["refined"],label="equation at observed states")
        a.set(xlabel="Time",ylabel=rf"$\dot x_{j}$",title=rf"Local rate of change in $x_{j}$")
    fig.legend(*ax[0].get_legend_handles_labels(),loc="outside lower center",ncol=2,frameon=False)
    return fig


def oscillation(case,equation):
    t=np.linspace(0,60,1201);x0=case["verification_initial_conditions"][0]
    p=w.rollout(equation,t,x0)
    if p is None:raise RuntimeError("No complete verification rollout.")
    # Long reference is illustrative. Acceptance targets still use observed t=8..20.
    truth=w.trajectory(w.vdp_rhs(case["mu"]),x0,t)["x"]
    tail=t>=30
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for states,color,label in [(truth,COLORS["data"],"reference"),(p,COLORS["refined"],"candidate")]:
        y=states[tail,0];tt=t[tail]
        ax[0].plot(tt,y,color=color,label=label,linestyle="-" if label=="reference" else "--")
        power=abs(np.fft.rfft(y-y.mean()))**2
        freq=np.fft.rfftfreq(len(y),d=t[1]-t[0])
        ax[1].plot(freq[1:],power[1:]/max(power[1:]),color=color,label=label)
    peaks,_=find_peaks(p[tail,0],prominence=.1*np.ptp(p[tail,0]))
    if len(peaks)>=2:
        tt=t[tail];yy=p[tail,0];i,j=peaks[:2]
        ax[0].scatter([tt[i],tt[j]],[yy[i],yy[j]],color=COLORS["refined"],s=30,zorder=5)
        low=min(truth[tail,0].min(),p[tail,0].min())
        high=max(truth[tail,0].max(),p[tail,0].max())
        span=high-low
        height=high+.18*span
        ax[0].annotate("",xy=(tt[i],height),xytext=(tt[j],height),
                       arrowprops={"arrowstyle":"<->", "color":COLORS["refined"]})
        ax[0].text((tt[i]+tt[j])/2,height+.04*span,"one period",ha="center",va="bottom",fontsize=9)
        ax[0].vlines([tt[i],tt[j]],[yy[i],yy[j]],height,colors=COLORS["refined"],linestyles=":",lw=.8)
        ax[0].set_ylim(low-.08*span,high+.5*span)
    ax[0].set(xlabel="Time",ylabel="$x_0$",title="Measure peak spacing after the transient")
    ax[1].set(xlabel="Frequency (cycles / time)",ylabel="Normalized power",xlim=(0,.8),title="Spectrum as a complementary view")
    fig.legend(*ax[0].get_legend_handles_labels(),loc="outside lower center",ncol=2,frameon=False)
    return fig


def comparison(case,models):
    tr=case["test"][models["test_index"]]
    fig,ax=plt.subplots(1,3,figsize=(12,3.2),layout="constrained")
    predictions=[("Data",tr["x"],COLORS["data"],"-"),
                 ("Single trajectory",w.rollout(models["baseline_raw"],tr["t"],tr["x0"]),COLORS["baseline"],"--"),
                 ("Pooled + verified",w.rollout(models["vg_raw"],tr["t"],tr["x0"]),COLORS["raw"],":"),
                 ("After refinement",w.rollout(models["vg_refined"],tr["t"],tr["x0"]),COLORS["refined"],"--")]
    for label,p,color,ls in predictions:
        if p is None:continue
        ax[0].plot(tr["t"],p[:,0],ls,color=color,label=label,lw=1.6)
        ax[1].plot(tr["t"],p[:,1],ls,color=color,lw=1.6)
        ax[2].plot(*p.T,ls,color=color,lw=1.6)
    ax[0].set(xlabel="Time",ylabel="$x_0$",title="Position-like coordinate")
    ax[1].set(xlabel="Time",ylabel="$x_1$",title="Velocity-like coordinate")
    ax[2].set(xlabel="$x_0$",ylabel="$x_1$",title="Phase portrait")
    fig.legend(*ax[0].get_legend_handles_labels(),loc="outside upper center",ncol=4,frameon=False)
    return fig


def alternative(case,good,bad):
    t=np.linspace(0,60,1201); x0=case["verification_initial_conditions"][0]
    fig,ax=plt.subplots(1,2,figsize=(10,3.1),layout="constrained")
    for label,eq,color in [("Selected equation",good,COLORS["refined"]),("Linear alternative",bad,COLORS["alternative"])]:
        p=w.rollout(eq,t,x0)
        if p is None:continue
        keep=t>=30
        ax[0].plot(t[keep],p[keep,0],label=label,color=color)
        ax[1].plot(*p[keep].T,label=label,color=color)
    ax[0].set(xlabel="Time",ylabel="$x_0$",title="Similar period; different amplitude")
    ax[1].set(xlabel="$x_0$",ylabel="$x_1$",title="Long-time orbit")
    fig.legend(*ax[0].get_legend_handles_labels(),loc="outside lower center",ncol=2,frameon=False)
    return fig


def cylinder_bridge():
    flow=w.load_cylinder()
    a=flow["artifact"]
    with np.load(w.CYLINDER/"reference_results/details/fixed_Re300_figure_payload.npz") as p:
        truth,pod,vg=[p[k][1].copy() for k in ["truth_images","pod_images","vg_images"]]
        extent=p["extent_star"].copy();limit=float(p["field_limit"])
    fig=plt.figure(figsize=(12,3.2),layout="constrained")
    gs=fig.add_gridspec(1,4,width_ratios=[1.2,1,1,1])
    ax=fig.add_subplot(gs[0])
    e=np.cumsum(a["singular_values"]**2)/float(a["total_training_fluctuation_energy"])
    ax.plot(np.arange(1,21),100*e,"o-",ms=3,color=COLORS["data"])
    ax.scatter([6],[100*e[5]],color=COLORS["refined"],s=55,zorder=3)
    ax.axvline(6,color=COLORS["refined"],alpha=.4,ls="--")
    ax.set(xlabel="Retained POD modes",ylabel="Fluctuation energy (%)",title="Representation choice",ylim=(35,102))
    for j,(field,title) in enumerate([(truth,"Simulation"),(pod,"Six-mode reconstruction"),(vg,"Symbolic prediction")],1):
        axy=fig.add_subplot(gs[j]);im=axy.imshow(field,origin="lower",extent=extent,cmap="RdBu_r",vmin=-limit,vmax=limit)
        axy.add_patch(Circle((0,0),.5,color="white",ec="black",lw=.8))
        axy.set(xlabel="$x/D$",ylabel="$y/D$",title=title)
    fig.colorbar(im,ax=fig.axes[1:],shrink=.65,label="Nondimensional vorticity")
    return fig


def flow_rollout(flow,prediction):
    fig,axes=plt.subplots(3,2,figsize=(10,6.4),layout="constrained")
    tr=flow["test"]
    for j,ax in enumerate(axes.flat):
        ax.plot(tr["t"],tr["x"][:,j],color=COLORS["data"],label="POD data")
        ax.plot(tr["t"],prediction[:,j],"--",color=COLORS["refined"],label="symbolic rollout")
        ax.set(xlabel="Time from test start",ylabel=rf"$z_{j}$")
    axes[0,0].legend(frameon=False)
    return fig


def rank_reconstruction(flow, ranks=(4,6,10,20), snapshot=100):
    a=flow["artifact"]
    coefficients=(a["singular_values"][:,None]*a["Vt_rank20"]).T
    fig,axes=plt.subplots(1,len(ranks),figsize=(12,3.1),layout="constrained")
    fig.suptitle(f"Same development snapshot: t = {50.1 + 0.1*snapshot:.1f}")
    fields=[]
    for rank in ranks:
        field=a["mean_field"][:,0]+a["U_rank20"][:,:rank]@coefficients[snapshot,:rank]
        fields.append(field.reshape((300,200), order="F").T)
    with np.load(w.ROOT/"data/cylinder_preview.npz") as payload:
        limit=float(payload["limit"])
    for ax,rank,field in zip(axes,ranks,fields):
        ax.imshow(field,origin="lower",extent=(-1,5,-2,2),cmap="RdBu_r",vmin=-limit,vmax=limit)
        ax.add_patch(Circle((0,0),.5,color="white",ec="black",lw=.8))
        ax.set(title=f"{rank} modes",xlabel="$x/D$",ylabel="$y/D$")
    return fig
