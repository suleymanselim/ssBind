#!/usr/bin/python
import os, math
import pandas as pd
import argparse
import itertools
from rdkit import Chem
from rdkit.Chem.rdMolAlign import AlignMol, GetBestRMS
from rdkit.Chem.rdmolops import Get3DDistanceMatrix
from rdkit.Chem.Draw import *
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolTransforms
from typing import Optional
from rdkit.Chem.rdmolfiles import * 
import multiprocessing as mp
from joblib import Parallel, delayed
from typing import List, Optional
from rdkit.Chem import AllChem
from copy import deepcopy
from plotnine import *
import patchworklib as pw
from rdkit.Chem import rdMolAlign

def ggplotheme():
	return theme(
	aspect_ratio = 2,
	panel_border = element_rect(color = "black", fill = 'NA', linewidth = 1.5), 
	panel_background = element_rect(fill = "None"),
	panel_grid_major = element_line(linewidth = 0),
    panel_grid_minor = element_line(linewidth = 0),
    axis_text = element_text(size = 16, color = "black", face = "bold"),
    axis_ticks = element_line(color = "black", linewidth = 2),
    axis_ticks_length = 5,
    axis_title_x = element_text(vjust = 1, size = 20, face = "bold"),
    axis_title_y = element_text(angle = 90, vjust = 1, size = 20, face = "bold"),
    legend_text = element_text(size = 10, face="bold"),
    legend_title = element_text(size = 16, hjust = 0, face="bold")
	)
    

def plotPCA(sdf, ref, output):
	import MDAnalysis as mda
	from MDAnalysis.analysis import pca, align

	confs = Chem.SDMolSupplier(sdf)

	u = mda.Universe(ref, confs)             
	pc = pca.PCA(u, select='all',
		         align=False, mean=None,
		         n_components=None).run()

	backbone = u.select_atoms('all')

	transformed = pc.transform(backbone, n_components=3)
	transformed.shape

	df = pd.DataFrame(transformed,
		              columns=['PC{}'.format(i+1) for i in range(3)])

	plot1 = (ggplot(df) + aes(x='PC1', y='PC2') + geom_pointdensity() + geom_density_2d(aes(color = '..level..'), size=1) + ggplotheme())
	plot2 = (ggplot(df) + aes(x='PC1', y='PC3') + geom_pointdensity() + geom_density_2d(aes(color = '..level..'), size=1) + ggplotheme())
	plot3 = (ggplot(df) + aes(x='PC2', y='PC3') + geom_pointdensity() + geom_density_2d(aes(color = '..level..'), size=1) + ggplotheme())
	g1 = pw.load_ggplot(plot1, figsize=(5,5))
	g2 = pw.load_ggplot(plot2, figsize=(5,5))
	g3 = pw.load_ggplot(plot3, figsize=(5,5))
	g1234 = (g1|g2|g3)
	g1234.savefig(output)



def cluster(sdf, ref, output):
	import MDAnalysis as mda
	from MDAnalysis.analysis import pca, align
	from rpy2 import robjects
	import rpy2.robjects.lib.ggplot2 as ggplot2

	confs = Chem.SDMolSupplier(sdf)

	u = mda.Universe(ref, confs)
	pc = pca.PCA(u, select='all',
		         align=False, mean=None,
		         n_components=None).run()

	backbone = u.select_atoms('all')

	transformed = pc.transform(backbone, n_components=3)
	transformed.shape

	df = pd.DataFrame(transformed,
		              columns=['PC{}'.format(i+1) for i in range(3)])


	df['Index'] = df.index * u.trajectory.dt

	liedata = pd.read_csv('LIE.csv', delimiter=',', header=0)
	PCA_LIE = pd.merge(df, liedata, on='Index')
	PCA_LIE.to_csv('PCA_LIE.csv', index=False, header=True)

	pcs = ['PC1', 'PC2', 'PC3']
	for i in range(len(pcs)):
		for j in range(i + 1, len(pcs)):

			index_data = []

			robjects.r(f'''
			library(ggplot2)

			df <- read.table('PCA_LIE.csv', header=TRUE, sep=",")

			p <- ggplot(df, aes(x={pcs[i]}, y={pcs[j]}, z=LIE)) +
				stat_summary_2d(fun=mean, binwidth = 0.25) 

			plot_data <- ggplot_build(p)
			bin_data <- plot_data$data[[1]]
			write.csv(bin_data, file = '{pcs[i]}_{pcs[j]}.csv', row.names = FALSE)
			''')

			data = pd.read_csv(f'{pcs[i]}_{pcs[j]}.csv', delimiter=',', header=0)
			raw_data = pd.read_csv('PCA_LIE.csv', delimiter=',', header=0)

			df_sorted = data.sort_values(by='value')
			top_bins = df_sorted.head(10)
			extracted_data = top_bins[['xmin', 'xmax','ymin', 'ymax']]

			for a, rowa in extracted_data.iterrows():
				for b, rowb in raw_data.iterrows():
					if rowa['xmin'] < rowb[pcs[i]] < rowa['xmax'] and rowa['ymin'] < rowb[pcs[j]] < rowa['ymax']:
						index_data.append(rowb)
			os.remove(f'{pcs[i]}_{pcs[j]}.csv')

			robjects.r(f'''
			library(ggplot2)

			mydata <- read.table('PCA_LIE.csv', header=TRUE, sep=",")

			p <- ggplot(mydata, aes(x={pcs[i]}, y={pcs[j]})) +
				stat_bin_2d(binwidth = 0.25, aes(fill = after_stat(density)))

			plot_data <- ggplot_build(p)
			bin_data <- plot_data$data[[1]]
			write.csv(bin_data, file = '{pcs[i]}_{pcs[j]}.csv', row.names = FALSE)
			''')

			data = pd.read_csv(f'{pcs[i]}_{pcs[j]}.csv', delimiter=',', header=0)
			raw_data = pd.read_csv('PCA_LIE.csv', delimiter=',', header=0)

			df_sorted = data.sort_values(by='density')
			top_bins = df_sorted.tail(10)
			extracted_data = top_bins[['xmin', 'xmax','ymin', 'ymax']]

			for a, rowa in extracted_data.iterrows():
				for b, rowb in raw_data.iterrows():
					if rowa['xmin'] < rowb[pcs[i]] < rowa['xmax'] and rowa['ymin'] < rowb[pcs[j]] < rowa['ymax']:
						index_data.append(rowb)
			os.remove(f'{pcs[i]}_{pcs[j]}.csv')
	


	cids = []
	for entry in index_data:
		cids.append(confs[int(entry['Index'])])

	dists = []
	for i in range(len(cids)):
		for j in range(i):
		    dists.append(rdMolAlign.GetBestRMS(cids[i],cids[j]))

	from rdkit.ML.Cluster import Butina
	clusts = Butina.ClusterData(dists, len(cids), 0.75, isDistData=True, reordering=True)
	PC1 = []
	PC2 = []
	PC3 = []
	outf = open('clusts.sdf','a')
	for i in clusts:
		a = index_data[i[0]]
		PC1.append(a['PC1'])
		PC2.append(a['PC2'])
		PC3.append(a['PC3'])
		sdwriter = Chem.SDWriter(outf)
		sdwriter.write(cids[i[0]])
		sdwriter.close()
	outf.close()
	

	pcx = [PC1, PC2, PC3]
	pcs = ['PC1', 'PC2', 'PC3']
	for i in range(len(pcs)):
		for j in range(i + 1, len(pcs)):

			robjects.r(f'''
			library(ggplot2)
			library(ggdensity)

			df <- read.table('PCA_LIE.csv', header=TRUE, sep=",")
			
			p <- ggplot(df, aes(x={pcs[i]}, y={pcs[j]}, z=LIE)) +
				stat_summary_2d(fun=mean, binwidth = 0.25) +
				scale_fill_gradientn(colours = rainbow(5), limits = c(min(df$LIE),max(df$LIE))) +
				geom_hdr_lines() +
				labs(fill = "LIE") +
				theme(
			panel.border = element_rect(color = "black", fill = 'NA', linewidth = 1.5),
			panel.background = element_rect(fill = NA),
			panel.grid.major = element_line(linewidth = 0),
			panel.grid.minor = element_line(linewidth = 0),
			axis.text = element_text(size = 16, color = "black", face = "bold"),
			axis.ticks = element_line(color = "black", linewidth = 1),
			axis.ticks.length = unit(5, "pt"),
			axis.title.x = element_text(vjust = 1, size = 20, face = "bold"),
			axis.title.y = element_text(angle = 90, vjust = 1, size = 20, face = "bold"),
			legend.text = element_text(size = 10, face="bold"),
			legend.title = element_text(size = 16, hjust = 0, face="bold")
			) +
			annotate("text", x = c(''' + str(', '.join(map(str, pcx[i]))) + '''), y = c(''' + str(', '.join(map(str, pcx[j]))) + '''), size = 6, 
			fontface = "bold", label = c(''' + str(', '.join(map(str, list(range(1, len(clusts) + 1))))) + f''')) +
			coord_fixed((max(df${pcs[i]})-min(df${pcs[i]}))/(max(df${pcs[j]})-min(df${pcs[j]})))

			ggsave("{pcs[i]}-{pcs[j]}_{output}", width = 7, height = 7)

			''')

 
