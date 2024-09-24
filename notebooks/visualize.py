from matplotlib.transforms import Affine2D
from matplotlib.spines import Spine
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.path import Path
from matplotlib.patches import Circle, RegularPolygon
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import contextily as ctx
import geopandas as gpd
from mycolorpy import colorlist as mcp


def vis_cluster_lineplots(data: pd.DataFrame, n_clusters: int, cluster_labels: np.ndarray, random_colors: bool = True, my_colors: list = []):
    if random_colors:
        my_colors = mcp.gen_color(cmap="tab20", n=n_clusters)

    ncols = 2
    # Calculate number of rows based on number of clusters
    nrows = int(np.ceil(n_clusters / ncols))
    # nrows =

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(
        3.25 * ncols, 2.5 * nrows), sharey=True, sharex=False)
    n_clusters_to_viz = n_clusters

    for i, label in enumerate(cluster_labels):
        # Visualise raw data
        ax[i // ncols, i % ncols].plot(data[data['label'] == label].iloc[:,
                                       :-1].T, color='gray', alpha=0.5, linewidth=0.5)

        # Get mean and 10th and 90th percentiles
        q1 = data[data['label'] == label].iloc[:, :-1].quantile(0.1, axis=0)
        mean = data[data['label'] == label].iloc[:, :-1].mean(axis=0)
        q3 = data[data['label'] == label].iloc[:, :-1].quantile(0.9, axis=0)

        # Visualise mean and 10th and 90th percentiles
        mean.plot(ax=ax[i // ncols, i % ncols],
                  color=my_colors[i], linewidth=3)
        q1.plot(ax=ax[i // ncols, i % ncols], color=my_colors[i],
                linewidth=0.9, linestyle='--')
        q3.plot(ax=ax[i // ncols, i % ncols], color=my_colors[i],
                linewidth=0.9, linestyle='--')

        # Add title with the number of data points
        ax[i // ncols, i %
            ncols].set_title(f'Cluster {label}, n={data[data["label"] == label].shape[0]}')

        # Remove spines
        ax[i // ncols, i % ncols].spines['right'].set_visible(False)
        ax[i // ncols, i % ncols].spines['top'].set_visible(False)

        # Set labels
        ax[i // ncols, i % ncols].set_ylabel('Ambulance calls (scaled)')
        ax[i // ncols, i % ncols].set_xlabel('Time of day (h)')

    # Construct legend
    handles, labels = ax[i // ncols, i % ncols].get_legend_handles_labels()
    handles = [matplotlib.lines.Line2D([0], [0], color='gray', linewidth=0.5),
               matplotlib.lines.Line2D(
                   [0], [0], color=my_colors[0], linewidth=3),
               matplotlib.lines.Line2D(
                   [0], [0], color=my_colors[0], linewidth=0.75, linestyle='--'),
               matplotlib.lines.Line2D([0], [0], color=my_colors[0], linewidth=0.75, linestyle='--')]
    labels = ['Data', 'Mean', '10th percentile', '90th percentile']

    # Add legend
    ax[0, 0].legend(handles, labels, frameon=False, loc='upper left')

    # Remove empty axes
    for i in range(n_clusters_to_viz, ncols * nrows):
        ax[i // ncols, i % ncols].axis('off')

    fig.tight_layout()


def vis_clusters_maps(data: pd.DataFrame, grid: gpd.GeoDataFrame, city_borders: gpd.GeoDataFrame, n_clusters: int, k_max: int, id_column: str = 'c28992r1000', random_colors: bool = True, my_colors: list = []):
    if random_colors:
        my_colors = mcp.gen_color(cmap="tab20", n=n_clusters)

    cities = ['Den Haag', 'Rotterdam', 'Amsterdam']
    for city in cities:
        my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", my_colors[:k_max])

        gdf = pd.merge(grid[[id_column, 'city', 'geometry']],
                       data, left_on=id_column, right_index=True)
        top_k_labels = gdf['label'].value_counts().index[:k_max]

        if city == 'Den Haag':
            municipality = "'s-Gravenhage"
        else:
            municipality = city

        ax = city_borders[city_borders['GM_NAAM'] == municipality].to_crs('EPSG:3857').plot(
            edgecolor='black', facecolor='none', linewidth=0.5, figsize=(8, 8))

        gdf = gdf[gdf['label'].isin(top_k_labels)]
        gdf[gdf['city'] == city].to_crs('EPSG:3857').plot(ax=ax,
                                                          column='label', categorical=True, legend=True, cmap=my_cmap, alpha=0.8, legend_kwds={'frameon': True, 'title': 'Cluster'})

        not_clustered = grid[grid['city'] == city]
        not_clustered = not_clustered[~not_clustered[id_column].isin(
            gdf[id_column])]
        # not_clustered.plot(ax=ax, edgecolor='black',
        #                    facecolor='none', linewidth=0.5)

        # Add a base map with contextily CartoDB positron
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

        # Save figure

        ax.axis('off')
        plt.savefig(f'../figures/results/{city}_clusters.png',
                    dpi=300, bbox_inches='tight')

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)

                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def plot_radarchart(data: pd.DataFrame, my_colors: list, agg_type: str,  quantile_range: tuple = (0.25, 0.75), fill_between: bool = False):
    if fill_between:
        cluster_label = data['label'].unique()[0]
    if agg_type == 'mean':
        aggs = data.groupby('label').mean().values
    elif agg_type == 'median':
        aggs = data.groupby('label').median().values
    else:
        raise ValueError(
            f'agg_type must be either "mean" or "median", not {agg_type}')

    Q1 = data.groupby('label').quantile(quantile_range[0]).values
    Q3 = data.groupby('label').quantile(quantile_range[1]).values
    N = len(data.columns[:-1])
    spoke_labels = data.columns[:-1]

    theta = radar_factory(N, frame='polygon')
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    # ax.set_rgrids([0.1, 0.3, 0.5])
    # ax.set_rgrids([0.1, 1, 5, 10, 15])
    # ax.set_yticklabels([])
    for agg, q1, q3, color in zip(aggs, Q1, Q3, my_colors):
        line = ax.plot(theta, agg, color=color, alpha=0.75, linewidth=2)
        if fill_between:
            # Connect the last and first point
            # ax.plot([theta[-1], theta[0]], [q3[-1], q3[0]], color=color,
            #         alpha=0.25, linewidth=2)
            # ax.plot([theta[-1], theta[0]], [q1[-1], q1[0]], color=color,
            #         alpha=0.25, linewidth=2)
            ax.fill_between(theta, q1, q3, color=color, alpha=0.25)
            # Fill between the first and the last point
            ax.fill_between([theta[-1], theta[0]], [q3[-1], q3[0]],
                            [q1[-1], q1[0]], color=color, alpha=0.25)

    ax.set_varlabels(spoke_labels)

    # Add legend for each cluster
    n_clusters = len(aggs)
    handles = []
    labels = []
    for i in range(n_clusters):
        handles.append(matplotlib.lines.Line2D(
            [0], [0], color=my_colors[i], linewidth=2))

        if fill_between:
            labels.append(f'{agg_type.capitalize()}')
        else:
            labels.append(f'Cluster {i}')

        # Add legend for fill between
        if fill_between:
            handles.append(matplotlib.patches.Patch(
                facecolor=my_colors[i], alpha=0.25))
            if fill_between:
                q1_str = int(quantile_range[0] * 100)
                q3_str = int(quantile_range[1] * 100)
                labels.append(f'{q1_str}th-{q3_str}th percentile')
    # Move legend to the right
    ax.legend(handles, labels, frameon=False, loc='upper right',
              bbox_to_anchor=(1.3, 1.1))
    # Remove the inner grid but keep only the outer grid
    # ax.grid(False)

    if fill_between:
        ax.set_title(f'Cluster {cluster_label}')
    return fig, ax
