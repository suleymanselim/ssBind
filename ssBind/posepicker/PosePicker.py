import logging
import os

import MDAnalysis as mda
import pandas as pd
from MDAnalysis.analysis import pca
from rdkit import Chem
from rdkit.ML.Cluster import Butina
from rpy2.robjects import pandas2ri, r

pandas2ri.activate()
import multiprocessing as mp
from contextlib import closing

import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper

from ssBind.io import get_model_compex

rpy2_logger.setLevel(logging.ERROR)


class PosePicker:
    def __init__(self, receptor_file: str, **kwargs) -> None:
        self._receptor_file = receptor_file
        self._binsize = kwargs.get("bin", 0.25)
        self._distThresh = kwargs.get("distThresh", 0.5)
        self._numbin = kwargs.get("numbin", 10)
        self._nprocs = kwargs.get("nprocs", 1)

    def pick_poses(
        self, conformers: str = "conformers.sdf", csv_scores: str = "Scores.csv"
    ) -> None:
        """
        Performs clustering analysis on the PCA-transformed data.

        """

        input_format = conformers.split(".")[-1].lower()

        flex = False
        if input_format == "pdb":
            confs = Chem.SDMolSupplier("conformers.sdf")
            u = mda.Universe(confs[0], confs)
            flex = True
        else:
            confs = Chem.SDMolSupplier(conformers, sanitize=False)
            u = mda.Universe(confs[0], confs)

        pc = pca.PCA(
            u, select="not (name H*)", align=False, mean=None, n_components=None
        ).run()

        atoms = u.select_atoms("not (name H*)")

        transformed = pc.transform(atoms, n_components=3)
        transformed.shape

        df = pd.DataFrame(transformed, columns=["PC{}".format(i + 1) for i in range(3)])

        df["Index"] = df.index * u.trajectory.dt

        scoredata = pd.read_csv(csv_scores, delimiter=",", header=0)
        PCA_Scores = pd.merge(df, scoredata, on="Index")
        PCA_Scores.to_csv("PCA_Scores.csv", index=False, header=True)

        pcs = ["PC1", "PC2", "PC3"]
        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):

                index_data = []

                r(
                    f"""
                library(ggplot2)

                df <- read.table('PCA_Scores.csv', header=TRUE, sep=",")

                p <- ggplot(df, aes(x={pcs[i]}, y={pcs[j]}, z=Score)) +
                    stat_summary_2d(fun=mean, binwidth = {self._binsize}) 

                plot_data <- ggplot_build(p)
                bin_data <- plot_data$data[[1]]
                write.csv(bin_data, file = '{pcs[i]}_{pcs[j]}.csv', row.names = FALSE)
                """
                )

                data = pd.read_csv(f"{pcs[i]}_{pcs[j]}.csv", delimiter=",", header=0)
                raw_data = pd.read_csv("PCA_Scores.csv", delimiter=",", header=0)

                df_sorted = data.sort_values(by="value")
                top_bins = df_sorted.head(self._numbin)
                extracted_data = top_bins[["xmin", "xmax", "ymin", "ymax"]]

                for a, rowa in extracted_data.iterrows():
                    for b, rowb in raw_data.iterrows():
                        if (
                            rowa["xmin"] < rowb[pcs[i]] < rowa["xmax"]
                            and rowa["ymin"] < rowb[pcs[j]] < rowa["ymax"]
                        ):
                            index_data.append(rowb)
                os.remove(f"{pcs[i]}_{pcs[j]}.csv")

                r(
                    f"""
                library(ggplot2)

                mydata <- read.table('PCA_Scores.csv', header=TRUE, sep=",")

                p <- ggplot(mydata, aes(x={pcs[i]}, y={pcs[j]})) +
                    stat_bin_2d(binwidth ={self._binsize}, aes(fill = after_stat(density)))

                plot_data <- ggplot_build(p)
                bin_data <- plot_data$data[[1]]
                write.csv(bin_data, file = '{pcs[i]}_{pcs[j]}.csv', row.names = FALSE)
                """
                )

                data = pd.read_csv(f"{pcs[i]}_{pcs[j]}.csv", delimiter=",", header=0)
                raw_data = pd.read_csv("PCA_Scores.csv", delimiter=",", header=0)

                df_sorted = data.sort_values(by="density")
                top_bins = df_sorted.tail(self._numbin)
                extracted_data = top_bins[["xmin", "xmax", "ymin", "ymax"]]

                for a, rowa in extracted_data.iterrows():
                    for b, rowb in raw_data.iterrows():
                        if (
                            rowa["xmin"] < rowb[pcs[i]] < rowa["xmax"]
                            and rowa["ymin"] < rowb[pcs[j]] < rowa["ymax"]
                        ):
                            if any(row["Index"] == rowb["Index"] for row in index_data):
                                index_data.append(rowb)
                os.remove(f"{pcs[i]}_{pcs[j]}.csv")

        cids = []
        index_dict = []

        for i, entry in enumerate(index_data):
            cids.append(confs[int(entry["Index"])])
            index_dict.append({i: int(entry["Index"])})

        dists = []
        tasks = [(i, j, cids[i], cids[j]) for i in range(len(cids)) for j in range(i)]
        with closing(mp.Pool(processes=self._nprocs)) as pool:
            results = pool.map(self._calculate_rms, tasks)

        dists = [rms[0] for _, _, rms in sorted(results, key=lambda x: (x[0], x[1]))]

        clusts = Butina.ClusterData(
            dists, len(cids), self._distThresh, isDistData=True, reordering=True
        )

        PC1 = []
        PC2 = []
        PC3 = []
        mode = 1
        for i in clusts:
            a = index_data[i[0]]
            PC1.append(a["PC1"])
            PC2.append(a["PC2"])
            PC3.append(a["PC3"])
            dict_i = index_dict[i[0]]

            if flex == True:
                model = int(next(iter(dict_i.values())))
                get_model_compex(
                    conformers, model, self._receptor_file, f"model_{mode}.pdb"
                )
            else:
                sdwriter = Chem.SDWriter(f"model_{mode}.sdf")
                sdwriter.write(confs[int(next(iter(dict_i.values())))])
                sdwriter.close()
            mode += 1

        pcx = [PC1, PC2, PC3]
        pcs = ["PC1", "PC2", "PC3"]
        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):

                r(
                    f"""
                library(ggplot2)
                library(ggdensity)

                df <- read.table('PCA_Scores.csv', header=TRUE, sep=",")
                
                p <- ggplot(df, aes(x={pcs[i]}, y={pcs[j]}, z=Score)) +
                    stat_summary_2d(fun=mean, binwidth = {self._binsize}) +
                    scale_fill_gradientn(colours = rainbow(5), limits = c(min(df$Score),max(df$Score))) +
                    geom_hdr_lines() +
                    labs(fill = "Score") +
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
                annotate("text", x = c("""
                    + str(", ".join(map(str, pcx[i])))
                    + """), y = c("""
                    + str(", ".join(map(str, pcx[j])))
                    + """), size = 6, 
                fontface = "bold", label = c("""
                    + str(", ".join(map(str, list(range(1, len(clusts) + 1)))))
                    + f""")) +
                coord_fixed((max(df${pcs[i]})-min(df${pcs[i]}))/(max(df${pcs[j]})-min(df${pcs[j]})))

                ggsave("{pcs[i]}-{pcs[j]}.svg", width = 7, height = 7)

                """
                )

    @staticmethod
    def _calculate_rms(params):
        i, j, cid_i, cid_j = params
        mol1 = Molecule.from_rdkit(cid_i)
        mol2 = Molecule.from_rdkit(cid_j)
        rms = rmsdwrapper(mol1, mol2)
        # rms = GetBestRMS(cid_i, cid_j)
        return i, j, rms
