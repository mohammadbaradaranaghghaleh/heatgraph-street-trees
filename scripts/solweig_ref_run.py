"""Run SOLWEIG-GPU (reference model) on prepared inputs. Executed INSIDE the `solweig-ref`
conda env (GDAL from conda-forge), not in HeatGraph/.venv:

    conda run -n solweig-ref python scripts/solweig_ref_run.py <input_dir> <YYYY-MM-DD>

<input_dir> holds Building_DSM.tif, DEM.tif, Trees.tif, metfile.txt (see heatgraph.realpatch).
Outputs: <input_dir>/output_folder/<tile>/TMRT_<tile>.tif (+ SVF, Shadow, UTCI).
"""
import os
import sys
import time

from solweig_gpu import thermal_comfort


def main(base, day):
    base = os.path.abspath(base)
    t = time.time()
    thermal_comfort(base, day, building_dsm_filename="Building_DSM.tif", dem_filename="DEM.tif",
                    trees_filename="Trees.tif", landcover_filename=None, ERA_5_z0_find=False,
                    tile_size=1000, overlap=20, use_own_met=True, own_met_file=os.path.join(base, "metfile.txt"),
                    use_uhi=False, save_tmrt=True, save_svf=True, save_shadow=True)
    print(f"[solweig_ref_run] done in {time.time() - t:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
