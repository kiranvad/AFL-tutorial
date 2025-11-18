import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import pathlib 
from matplotlib import gridspec
from scipy.stats import gaussian_kde
from matplotlib.ticker import ScalarFormatter
import matplotlib.transforms as mtransforms 
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap

from AFL.double_agent import *
from VirtualSAS_theory import VirtualSAS_theory
from AFL.automation.APIServer.data.DataTrashcan import DataTrashcan

def get_simulator(dir):
    boundary_dataset = xr.load_dataset(dir+'/250610-extrap_expand_dataset.nc')
    boundary_dataset.close()

    reference_data_path = dir+'/SANS/'
    inst_client = VirtualSAS_theory()
    inst_client.data = DataTrashcan()
    boundary_dataset['labels'] = boundary_dataset.labels.astype(str)
    inst_client.boundary_dataset = boundary_dataset
    inst_client.trace_boundaries(hull_tracing_ratio=0.25)

    # specify reference data
    for fname in ['low_q.ABS', 'med_q.ABS', 'high_q.ABS']:
        data = pd.read_csv(str(pathlib.Path(reference_data_path) / fname),sep=r'\s+')
        inst_client.add_configuration(
            q=list(data.q),
            I=list(data.I),
            dI=list(data.dI),
            dq=list(data.dq),
            reset=False
        )

    inst_client.add_sasview_model(
        label='2',
        model_name='polymer_excl_volume',
        model_kw={
            'scale': 1.0,
            'background': 1.0,
            'rg': 100.0,
        }
    )

    inst_client.add_sasview_model(
        label='1',
        model_name='sphere',
        model_kw={
            'scale': 0.5,
            'background': 1.0,
            'sld': 1.0,
            'sld_solvent': 6.0,
            'radius': 10,
        }
    )

    inst_client.add_sasview_model(
        label='0',
        model_name='power_law',
        model_kw={
            'scale': 1e-7,
            'background': 1.0,
            'power': 4.0,
        }
    ) 

    return inst_client

def plot_phase_diagram(ds: xr.Dataset, grid_variable : str, dim : str, labels_variable : str, ax = None):
    phase_colors =['tab:red','tab:olive','tab:cyan']
    cmap = ListedColormap(phase_colors)
    boundaries = np.linspace(1,4,4)
    norm = BoundaryNorm(boundaries, cmap.N)
    if ax==None:
        fig, ax  = plt.subplots(subplot_kw={"projection":'3d'})
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 12)
        ax.set_zlim(0, 50)
        ax.set_xlabel("protein")
        ax.set_ylabel("glycerol")
        ax.set_zlabel("temperature", labelpad=-1)
    else:
        fig = plt.gcf()

    xg = ds[grid_variable].sel({dim : 'protein'}).values
    yg = ds[grid_variable].sel({dim : 'glycerol'}).values
    zg = ds[grid_variable].sel({dim : 'temperature'}).values
    for l in np.unique(ds[labels_variable]):
        flags = ds[labels_variable]==l
        ax.scatter(
            xg[flags], 
            yg[flags], 
            zg[flags],
            color=phase_colors[l],
            s=20
        )

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cax = ax.inset_axes([1.15, 0.15, 0.05, 0.7], transform=ax.transAxes)
    cbar = fig.colorbar(mappable, pad = 0.5, shrink=0.75, aspect=5, ticks=[1.5,2.5,3.5],cax=cax)
    cbar.ax.set_yticklabels(['S-L', 'Single', 'L-L'])
    
    return ax 

def plot_progress_p1(
        ds, 
        design_space_variable="design_space", 
        grid_variable="design_space_grid", 
        design_space_dim="ds_dim", 
        composition_variables=["protein", "glycerol"],
        temperature_variable="temperature",
        labels_variable="phases",
        phase_probability_variable="phase_entropy",
        phase_label_variable="phase_mean"
    ):
    """Plot cost and entropy in 3D."""
    fig = plt.figure(figsize=(4*4, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 0.05, 1], wspace=1e-5)

    ax_sampling = fig.add_subplot(gs[0, 0], projection="3d")
    x = ds[design_space_variable].sel({design_space_dim:composition_variables[0]}).values
    y = ds[design_space_variable].sel({design_space_dim:composition_variables[1]}).values
    z = ds[design_space_variable].sel({design_space_dim:temperature_variable}).values
    ax_sampling.scatter(x, y, z, c=ds[labels_variable].values)

    xg = ds[grid_variable].sel({design_space_dim:composition_variables[0]}).values
    yg = ds[grid_variable].sel({design_space_dim:composition_variables[1]}).values
    zg = ds[grid_variable].sel({design_space_dim:temperature_variable}).values    

    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    cmap = plt.get_cmap("coolwarm")
    ax_entropy = fig.add_subplot(gs[0, 1], projection="3d")
    mappable = ax_entropy.scatter(xg, yg, zg,
        c=ds[phase_probability_variable].values,
        cmap=cmap,
        norm = norm,
        s=20
    )
    cax = fig.add_subplot(gs[0, 2])
    cax.set_position([0.63, 0.25, 0.01, 0.5])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(phase_probability_variable)

    ax_pd = fig.add_subplot(gs[0, 3], projection="3d")
    plot_phase_diagram(
        ds, 
        grid_variable,
        design_space_dim,
        phase_label_variable, 
        ax = ax_pd
    )
    for ax in [ax_sampling, ax_entropy, ax_pd]:
        ax.set_xlabel(composition_variables[0])
        ax.set_ylabel(composition_variables[1])
        ax.set_zlabel(temperature_variable, labelpad=1)
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 12)
        ax.set_zlim(0, 50)
    
    
    return fig, [ax_sampling, ax_entropy, cax, ax_pd] 

def plot_progress(ds, 
                  design_space_variable="design_space", 
                  grid_variable="design_space_grid", 
                  design_space_dim="ds_dim", 
                  composition_variables=["protein", "glycerol"],
                  temperature_variable="temperature",
                  labels_variable="phases",
                  phase_probability_variable="phase_entropy",
                  feasibilty_probability_variable="feasibility_y_prob",
                  phase_label_variable="phase_mean"
                  ):
    """Plot cost and entropy in 3D."""
    fig, axs  = plt.subplots(1,4, 
                             figsize=(4*4, 4*1), 
                             subplot_kw={"projection":'3d'},
                             )
    axs = axs.flatten()
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    ax = axs[0]
    x = ds[design_space_variable].sel({design_space_dim:composition_variables[0]}).values
    y = ds[design_space_variable].sel({design_space_dim:composition_variables[1]}).values
    z = ds[design_space_variable].sel({design_space_dim:temperature_variable}).values
    ax.scatter(x, y, z, c=ds[labels_variable].values)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 12)
    ax.set_zlim(0, 50)

    xg = ds[grid_variable].sel({design_space_dim:composition_variables[0]}).values
    yg = ds[grid_variable].sel({design_space_dim:composition_variables[1]}).values
    zg = ds[grid_variable].sel({design_space_dim:temperature_variable}).values    

    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    cmap = plt.get_cmap("coolwarm")
    ax = axs[1]
    mappable = ax.scatter(xg, yg, zg,
        c=ds[phase_probability_variable].values,
        cmap=cmap,
        norm = norm,
        s=20
    )
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.15)
    cbar.set_label(phase_probability_variable)

    ax = axs[2]
    mappable = ax.scatter(xg, yg, zg,
        c=ds[feasibilty_probability_variable].values[:,1],
        cmap=cmap,
        norm = norm,
        s=20
    )
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.15)
    cbar.set_label(feasibilty_probability_variable)

    ax = axs[-1]
    plot_phase_diagram(
        ds,
        grid_variable, 
        design_space_dim, 
        phase_label_variable, 
        ax = ax
        )

    for ax in axs:
        ax.set_xlabel(composition_variables[0])
        ax.set_ylabel(composition_variables[1])
        ax.set_zlabel(temperature_variable, labelpad=-2)
    plt.tight_layout()

    return fig, axs 

def plot_sampling(ds,
                  design_space,
                  design_space_dim,
                  composition_sampling_domain,
                  composition_sampling_z,
                  composition_next,
                  temperature_domain,
                  temperature_y,
                  temperature_next
                  ):
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(nrows=2, 
                        ncols=2, 
                        height_ratios=[0.05, 0.95], 
                        hspace=0.05, wspace=0.25
                    )

    ax_cb = fig.add_subplot(gs[0, 0])       # colorbar axis above left plot
    ax_comp = fig.add_subplot(gs[1, 0])      # 2D KDE plot
    ax_temp = fig.add_subplot(gs[1, 1])     # 1D KDE plot

    ux = ds[composition_sampling_z].values
    x = ds[composition_sampling_domain].values
    mappable = ax_comp.tricontourf(x[:,0], x[:,1], ux)
    cbar = fig.colorbar(mappable, cax=ax_cb, orientation='horizontal')
    cbar.set_label(r"$\sum_{t}u(x,t)$", labelpad=15)
    ax_cb.xaxis.set_ticks_position('top')
    ax_cb.xaxis.set_label_position('top')

    ax_comp.scatter(ds[design_space].sel({design_space_dim:"protein"}).values ,
            ds[design_space].sel({design_space_dim:"glycerol"}).values,
            color = "k"
        )
    ax_comp.scatter(ds[composition_next].sel({"protein_glycerol_d":"protein"}).values ,
            ds[composition_next].sel({"protein_glycerol_d":"glycerol"}).values,
            color = "tab:red",
        )
    ax_comp.set_xlim([0.0, 80.0])
    ax_comp.set_ylim([0.0, 12.0])
    ax_comp.set_xlabel("protein")
    ax_comp.set_ylabel("glycerol")

    ax_temp.plot(
        ds[temperature_domain].values,
        ds[temperature_y].values,
        )
    t_opt = ds[temperature_next].values 
    for ti in t_opt:
        ax_temp.axvline(ti, ls='--', color="k")
    ax_temp.set_xlabel("Temperature")
    ax_temp.set_ylabel(r"$v(t)$")
        
    return fig, [ax_cb, ax_comp, ax_temp] 

def plot_sampling_densities(ds,
                          design_space_variable = "design_space",
                          design_space_dim = "ds_dim",
                          composition_variables = ["protein", "glycerol"],
                          temperature_variable = "temperature"
                        ):
    x = ds[design_space_variable].sel({design_space_dim: composition_variables[0]}).values
    y = ds[design_space_variable].sel({design_space_dim: composition_variables[1]}).values
    z = ds[design_space_variable].sel({design_space_dim: temperature_variable}).values

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(nrows=2, 
                        ncols=2, 
                        height_ratios=[0.05, 0.95], 
                        hspace=0.05, wspace=0.35)

    ax_cb = fig.add_subplot(gs[0, 0])       # colorbar axis above left plot
    ax_kde = fig.add_subplot(gs[1, 0])      # 2D KDE plot
    ax_hist = fig.add_subplot(gs[1, 1])     # 1D KDE plot

    # --- 2D KDE of (x, y) ---
    xy = np.vstack([x, y])

    # Grid
    xgrid = np.linspace(0, 80, 100)
    ygrid = np.linspace(0, 12, 100)
    X, Y = np.meshgrid(xgrid, ygrid)
    # Plot 2D density
    positions = np.vstack([X.ravel(), Y.ravel()])
    kde_xy = gaussian_kde(xy, bw_method=0.5)
    Z = np.reshape(kde_xy(positions).T, X.shape)
    cf = ax_kde.contourf(X, Y, Z, levels=20, cmap='Blues')
    ax_kde.scatter(x,y, color="k")
    ax_kde.set_xlabel(composition_variables[0])
    ax_kde.set_ylabel(composition_variables[1])
    ax_kde.set_xlim([0.0, 80.0])
    ax_kde.set_ylim([0.0, 12.0])

    # Add horizontal colorbar above ax_kde
    norm = Normalize(vmin=Z.min(), vmax=Z.max())
    sm = ScalarMappable(norm=norm, cmap='Blues')
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cbar.set_label("Density", labelpad=5)
    ax_cb.xaxis.set_ticks_position('top')
    ax_cb.xaxis.set_label_position('top')

    # Use scientific notation with offset (e.g., 1e-3)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))  # force sci notation if outside range
    ax_cb.xaxis.set_major_formatter(formatter)
    offset_text = ax_cb.xaxis.get_offset_text()
    # Anchor left and top as before
    offset_text.set_horizontalalignment('left')
    offset_text.set_verticalalignment('top')
    # Remove the default transform and set a new one with an offset in display coords
    trans = offset_text.get_transform()  # usually data or axes coords
    # Add an offset in display coords (pixels): x=+20 pixels, y=-10 pixels (down)
    offset = mtransforms.ScaledTranslation(5/72, -18/72, fig.dpi_scale_trans)
    offset_text.set_transform(trans + offset) 


    # Remove cb axis border for aesthetics
    for spine in ax_cb.spines.values():
        spine.set_visible(False)

    # --- 1D KDE of z ---
    kde_z = gaussian_kde(z, bw_method=0.5)
    zgrid = np.linspace(z.min(), z.max(), 200)
    density = kde_z(zgrid)

    ax_hist.fill_between(zgrid, density, color="tab:blue", alpha=0.6)
    ax_hist.plot(zgrid, density, color="tab:blue", lw=2.0)
    ax_hist.set_xlabel(temperature_variable)
    ax_hist.set_ylabel("Density")

    return fig, [ax_cb, ax_hist, ax_kde]

def _plot_temperature_profies(ds,
                          design_space_variable = "design_space",
                          design_space_dim = "ds_dim",
                          composition_variables = ["protein", "glycerol"],
                          temperature_variable = "temperature",
                          verbose = False,
                          ax = None
                        ):
    # convert to DataFrame
    df = ds[design_space_variable].to_dataframe("value").unstack(design_space_dim)["value"].reset_index()
    df.columns = ["sample", "protein", "temperature", "glycerol"]

    # make the (protein, glycerol) key
    df["pair"] = list(zip(df[composition_variables[0]], df[composition_variables[1]]))

    # detect when a new batch starts:
    # whenever the pair changes OR sample is not consecutive
    df["batch_id"] = (
        (df["pair"] != df["pair"].shift()) | (df["sample"] != df["sample"].shift() + 1)
    ).cumsum()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=1, vmax=len(df["batch_id"].unique()))
    for ind, rows in df.groupby("batch_id"):
        v = rows[temperature_variable].values
        k = rows["pair"].unique()[0]
        if verbose:
            print(
                f"{ind}\t"
                f"{'(' + ', '.join(f'{x:.2f}' for x in k) + ')'}\t"
                f"{'(' + ', '.join(f'{x:.2f}' for x in v) + ')'}"
            )

        ax.plot(
            np.linspace(0, 1, len(v)), 
            v, 
            color=cmap(norm(ind)), 
        )
    ax.set_xlabel("Measurement Index")
    ax.set_ylabel("Temperature")
    ax.set_ylim([0.0, 50.0])
    ax.set_xlim([0.0, 1.0])
    ax.plot(
        np.linspace(0.0, 1.0, 10), 
        np.linspace(0.0, 50.0, 10), 
        color="k", 
        ls="--"
    )
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cax = ax.inset_axes([1.15, 0.15, 0.025, 0.7], transform=ax.transAxes)
    cbar = fig.colorbar(mappable, pad = 0.5, shrink=0.85, aspect=5,cax=cax)
    cbar.set_label("Measurement Batch Index")

    return df, fig, ax, cbar

def plot_temperature_profies(ds,
                          design_space_variable = "design_space",
                          design_space_dim = "ds_dim",
                          batch_variable = "batch_sample_id",
                          composition_variables = ["protein", "glycerol"],
                          temperature_variable = "temperature",
                          ax = None
                        ):
    cmap = plt.get_cmap('coolwarm')
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    design_space = ds[design_space_variable]
    temperature_design_space = design_space.sel({design_space_dim:temperature_variable})
    batches = np.unique(ds[batch_variable])
    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=1, vmax=len(batches))
    for ind in batches:
        v = temperature_design_space[ds[batch_variable]==ind].values
        if len(v)>1:
            ax.plot(
                np.linspace(0, 1, len(v)), 
                v, 
                color=cmap(norm(ind)), 
                marker="o"
            )
        else:
            ax.scatter(0.0, v.item(), color=cmap(norm(ind)), s=50)
    ax.set_xlabel("Measurement Index")
    ax.set_ylabel("Temperature")
    ax.set_ylim([0.0, 50.0])
    ax.set_xlim([0.0, 1.0])
    ax.plot(
        np.linspace(0.0, 1.0, 10), 
        np.linspace(0.0, 50.0, 10), 
        color="k", 
        ls="--"
    )
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cax = ax.inset_axes([1.05, 0.15, 0.025, 0.7], transform=ax.transAxes)
    cbar = fig.colorbar(mappable, pad = 0.15, shrink=0.85, aspect=5,cax=cax)
    cbar.set_label("Measurement Batch Index")
    return [], fig, ax, cbar