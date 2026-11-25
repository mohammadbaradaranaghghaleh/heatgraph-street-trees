"""Block C: pure-Python 2.5D radiation + UTCI engine (label generator for the surrogate).

Pipeline per patch and hour (all rasters on the same 1 m working grid):

    sun.positions()           pvlib NREL SPA -> solar azimuth/elevation per hour
    surfaces.build_grids()    buildings.gpkg + canopy -> DSM, CDSM, TDSM, wall height/aspect, masks
    shadow.cast()             shifted-max raster shadow casting (Ratti & Richens 2004), trees with
                              transmissivity -> sunlit fraction (0..1) per pixel
    svf.compute()             horizon-angle SVF, split into building (SVF_b) and vegetation (SVF_v)
    radiation.tmrt()          six-directional short/long-wave fluxes (Thorsson et al. 2007;
                              Lindberg et al. 2008 formulation) -> Tmrt
    comfort.utci_grid()       pythermalcomfort UTCI from Ta, Tmrt, v(1.1 m), RH
    receptors.sample()        UTCI at pedestrian receptor points -> per-edge statistics

Static terms (SVF, wall geometry) are computed once per patch; only shadows and fluxes
are recomputed per hour. Validation target: Tmrt MAE <= engine.toml[validation] versus
solweig-gpu on a reference subset (Block I), reported before any surrogate training.
"""
