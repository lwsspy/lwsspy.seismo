import lwsspy as lpy
# Load Catalogs
initcat = lpy.seismo.CMTsmo.CMTCatalog.load("initcat.pkl")
g3dcat = lpy.seismo.CMTsmo.CMTCatalog.load("g3dcat.pkl")

# Create comparison class
C = lpy.seismo.CompareCatalogs(
    initcat, g3dcat, oldlabel='GCMT', newlabel='GCMT3D+')
Cfilt = C.filter(maxdict=dict(depth_in_m=30000.0, M0=.5))
C.plot_eps_nu()

# Make sure base is unique
unicat = initcat.unique(ret=True)
oldcat, newcat = unicat.check_ids(g3dcat)

# Get data to plot events
N = len(oldcat)
latitude = unicat.getvals("latitude")
longitude = unicat.getvals("longitude")
moment = unicat.getvals("moment_magnitude")
depth = unicat.getvals("depth_in_m")/1000.0

# Plot events
scatter, ax, l1, l2 = plot_quakes(latitude, longitude, depth, moment)
ax.set_global()
fig = ax.figure
fig.set_size_inches(6, 4)
plot_label(ax, f"N: {N}", location=1, box=False, dist=0.0)
savefig("inverted_events.pdf")

# Plot Moment tensor source-type dist
old_eps_nu = oldcat.getvals("decomp", "eps_nu")
new_eps_nu = newcat.getvals("decomp", "eps_nu")

# Compute difference
deps = new_eps_nu[:, 0] - old_eps_nu[:, 0]

# Plot
fig = figure(figsize=(12, 5))

# Plot histogram GCMT
ax = subplot(121)
bins = arange(-0.5, 0.50001, 0.01)
hist(old_eps_nu[:, 0], bins=bins, edgecolor='k',
     facecolor=(0.3, 0.3, 0.8, 0.75), linewidth=0.75,
     label=self.oldlabel, histtype='stepfilled')
# Plot histogram GCMT3D+
hist(new_eps_nu[:, 0], bins=bins, edgecolor='k',
     facecolor=(0.3, 0.8, 0.3, 0.75), linewidth=0.75,
     label=self.newlabel, histtype='stepfilled')
legend(loc='upper left', frameon=False, fancybox=False,
       numpoints=1, scatterpoints=1, fontsize='x-small',
       borderaxespad=0.0, borderpad=0.5, handletextpad=0.2,
       labelspacing=0.2, handlelength=1.0,
       bbox_to_anchor=(0.0, 1.0))

plot_label(ax, f"\n$\\mu$ = {np.mean(old_eps_nu[:,0]):7.4f}\n"
               f"$\\sigma$ = {np.std(old_eps_nu[:,0]):7.4f}\n"
               f"GCMT3D+\n$\\mu$ = {np.mean(new_eps_nu[:,0]):7.4f}\n"
               f"$\\sigma$ = {np.std(new_eps_nu[:,0]):7.4f}\n",
               location=2, box=False, fontdict=dict(fontsize='small', fontfamily="monospace"))
plot_label(ax, "CLVD-", location=6, box=False, fontdict=dict(fontsize='small'))
plot_label(ax, "CLVD+", location=7, box=False, fontdict=dict(fontsize='small'))
plot_label(ax, "DC", location=14, box=False, fontdict=dict(fontsize='small'))
xlabel(r"$\epsilon$")


ax = subplot(122)
bins = arange(-0.25, 0.25001, 0.005)
hist(deps, bins=bins, edgecolor='k',
     facecolor=(0.8, 0.8, 0.8, 0.75), linewidth=0.75,
     histtype='stepfilled')
xlabel(r"$\Delta\epsilon$")
plot_label(ax, f"$\\mu$ = {np.mean(deps):7.4f}\n"
               f"$\\sigma$ = {np.std(deps):7.4f}\n",
               location=2, box=False, fontdict=dict(fontsize='small', fontfamily="monospace"))
