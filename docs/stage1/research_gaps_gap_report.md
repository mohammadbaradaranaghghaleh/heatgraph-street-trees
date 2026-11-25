
# Research Gap Analysis Report

Generated: 2026-11-24 23:19

## Portfolio Summary

- **Total gaps identified**: 4
- **Valid gaps**: 4/4
- **High-significance gaps**: 3
- **Average evidence per gap**: 6.0
- **Quality**: ✓ PASS

## Gap Type Distribution

- Complete gaps (unexplored): 1
- Partial gaps (understudied): 3
- Controversy gaps (unresolved): 0

## Portfolio Assessment

✓ Gap portfolio meets quality standards

## Detailed Gap Analysis

### ✓ Gap 1: Thermal-comfort surrogates are validated on held-out scenes, not on the edits an optimiser makes

**ID**: GAP-001

**Type**: partial

**Significance**: high

**Feasibility**: High. The pure-Python engine can generate counterfactual edits cheaply (about 50 per patch), split-conformal intervals need no extra training, and re-simulation of 10-20 Pareto designs is a bounded CPU cost.

**Definition**:
Neural surrogates of Tmrt and urban microclimate are trained and tested on observed or randomly generated urban scenes, and their error is reported as an average over held-out areas and days. When the same surrogates drive tree-placement or roof-allocation searches, the optimiser evaluates thousands of edited scenes that differ systematically from the training distribution. No reviewed study reports surrogate error on intervention edits, carries calibrated uncertainty into the objective, or re-simulates the final optimised designs with the physical model.

**Evidence** (6 pieces):

1. **Briegel et al. (2023), Urban Climate, doi:10.1016/j.uclim.2022.101359**
   > "[paraphrase] A U-Net emulating SOLWEIG Tmrt was validated against SOLWEIG on 6 areas and 12 days (MAE about 2.4 K) and ran about 22 times faster."
   - Context: Validation is on held-out areas/days of observed scenes; no evaluation on counterfactual tree edits.

2. **Briegel et al. (2024), Geoscientific Model Development, doi:10.5194/gmd-17-1667-2024**
   > "[paraphrase] The same U-Net is reused as the Tmrt component of a larger machine-learning emulator for city-wide multi-scale thermal comfort."
   - Context: Surrogate used for long-term mapping, not for decision optimisation; error under edits not reported.

3. **Schrodi et al. (2023), arXiv, doi:10.48550/arXiv.2310.05691**
   > "[paraphrase] A neural Tmrt approximation makes optimisation of tree placements over large areas tractable, using iterated local search."
   - Context: Surrogate drives the search; whether reported gains hold when the optimised layouts are re-simulated must be checked in the full text [to verify].

4. **Kim et al. (2025), Journal of Environmental Management, doi:10.1016/j.jenvman.2025.125398**
   > "[paraphrase] A deep-learning surrogate with Pareto optimisation allocates cool and green roofs across urban grids under an end-of-century climate scenario."
   - Context: Adjacent domain (roofs); surrogate-in-the-loop optimisation without reported conformal uncertainty in the objective [to verify].

5. **Xin et al. (2025), UrbanGraph, ICLR 2026, doi:10.48550/arXiv.2510.00457**
   > "[paraphrase] A physics-informed dynamic heterogeneous graph encodes shading and convection in its structure and is evaluated on a simulated multi-configuration microclimate benchmark."
   - Context: Prediction benchmark; no intervention/optimisation evaluation.

6. **Yu et al. (2024), Urban Climate, doi:10.1016/j.uclim.2024.102003**
   > "[paraphrase] Graph neural networks estimate street-level air temperature more consistently across test locations than other data-driven approaches."
   - Context: Graph models are viable for street-level prediction, but evaluated as estimators, not as decision surrogates.

**Why This Matters**:
Planners use surrogate-optimised layouts as recommendations, so the accuracy that matters is accuracy on the designs the optimiser selects, not average accuracy on observed scenes. Optimisers actively search for inputs where a surrogate predicts unusually low heat, which is exactly where surrogate error is likely to be largest; without testing on edits, reported cooling gains may be partly artefacts of model error. The published surrogates for Tmrt (Briegel et al., 2023, 2024) and the surrogate-driven placement and roof-allocation studies (Schrodi et al., 2023; Kim et al., 2025) report held-out-scene accuracy and do not, as far as reviewed, quantify the surrogate-optimality gap or carry calibrated uncertainty into the decision. Graph surrogates (UrbanGraph; Yu et al., 2024) show that the representation works, which makes the question of decision-grade reliability the next step rather than a speculative one. Closing this gap gives a reusable protocol (counterfactual test set, conformal objective, re-simulation) that applies beyond trees to any surrogate-driven urban design optimisation.

---

### ✓ Gap 2: Tree-placement objectives measure heat where it is, not where people walk

**ID**: GAP-002

**Type**: complete

**Significance**: high

**Feasibility**: High. OSM networks and destinations are open; a Sevtsuk-style flow model and population weights (WorldPop) are computable on CPU; the objective change is a weighting of per-edge UTCI already produced by the engine.

**Definition**:
Tree-placement optimisation studies minimise area-averaged or pixel-level Tmrt or UTCI, treating every square metre as equally important. Separately, pedestrian-flow and heat-aware routing studies model where people walk and how much sun they receive, but take the urban form as fixed. No reviewed study optimises tree locations against a person-weighted heat dose along the pedestrian network, so the people who actually experience heat are absent from the objective that places trees.

**Evidence** (6 pieces):

1. **Schrodi et al. (2023), arXiv, doi:10.48550/arXiv.2310.05691**
   > "[paraphrase] The authors suggest that future work could integrate pedestrian density maps into the placement objective."
   - Context: The leading surrogate-based placement study names person-weighting as an open extension.

2. **Lachapelle et al. (2023), Landscape and Urban Planning, doi:10.1016/j.landurbplan.2022.104608**
   > "[paraphrase] A tree planted on a street with no existing shade gives roughly 1.5-2 times the pedestrian radiative cooling of one planted where the sidewalk is already shaded."
   - Context: Pedestrian benefit depends strongly on location and context, so the objective's spatial weighting changes the optimum.

3. **Kolaxidis et al. (2025), Transactions in GIS, doi:10.1111/tgis.70110**
   > "[paraphrase] Shade-aware pedestrian routing reduced solar exposure by about 12-26% in most cases, and network analysis flagged roads where added shade would matter most."
   - Context: Routing literature identifies priority streets but does not optimise the interventions.

4. **Vu (2025), arXiv, doi:10.48550/arXiv.2512.11896**
   > "[paraphrase] Land surface temperature predicted from street-view imagery is mapped onto pedestrian network nodes to enable heat-aware routing with an environmental-justice framing."
   - Context: Network-level exposure without an intervention model; uses LST rather than a comfort index.

5. **Sevtsuk (2021), Journal of the American Planning Association, doi:10.1080/01944363.2020.1864758**
   > "[paraphrase] Pedestrian flows on every street segment can be estimated by extending betweenness to origin-destination trips with destination attractiveness and bounded detours, calibrated with counts."
   - Context: Provides the open method to weight exposure by expected foot traffic.

6. **Kuras et al. (2017), Environmental Health Perspectives, doi:10.1289/ehp556**
   > "[paraphrase] Personal heat exposure can differ substantially from ambient or fixed-station estimates, motivating person-level exposure metrics."
   - Context: Supports moving the objective from area-averaged heat to experienced exposure.

**Why This Matters**:
Street trees are planted along sidewalks to protect people, yet the optimisation literature scores layouts by heat averaged over space, so a tree that shades an empty service road can count as much as one that shades a busy route to a school or transit stop. Evidence that pedestrian cooling per tree depends strongly on existing shade (Lachapelle et al., 2023) and that routing studies can already identify streets where shade matters most (Kolaxidis et al., 2025) shows that the spatial weighting is not a detail: it changes which locations are optimal. The tools to weight by expected foot traffic exist and are validated (Sevtsuk, 2021), and exposure science argues for person-level metrics (Kuras et al., 2017). Combining them answers a planning question the current literature cannot: how many person-hours of heat stress a given tree budget avoids, and whether area-optimal and person-optimal layouts differ.

---

### ✓ Gap 3: Heat and canopy inequity is well documented but not built into placement decisions

**ID**: GAP-003

**Type**: partial

**Significance**: high

**Feasibility**: Medium-high. Theil/Gini of dose across neighbourhoods is a cheap objective for NSGA-III; vulnerability weights from WorldPop age structures are open. The European setting differs from the US-dominated equity literature, which must be acknowledged.

**Definition**:
A large body of work shows that lower-income and minority neighbourhoods have less tree canopy and higher heat exposure, across many cities and with historical roots in housing policy. These studies are descriptive or associational. Tree-placement and greening-allocation optimisation, in turn, maximises total cooling and reports distributional effects, if at all, after the fact. Equity is rarely a formal objective or constraint in the optimiser, so the trade-off between total benefit and fair distribution is not quantified.

**Evidence** (6 pieces):

1. **McDonald et al. (2021), PLOS ONE, doi:10.1371/journal.pone.0249715**
   > "[paraphrase] Across 5,723 US communities, low-income blocks have about 15% less tree cover and are about 1.5 C hotter than high-income blocks."
   - Context: Large-sample descriptive evidence of canopy and heat disparity.

2. **Hsu et al. (2021), Nature Communications, doi:10.1038/s41467-021-22799-5**
   > "[paraphrase] Lower-income residents and people of colour are exposed to higher urban heat island intensity in most large US cities."
   - Context: Exposure disparity documented; no intervention allocation.

3. **Locke et al. (2021), npj Urban Sustainability, doi:10.1038/s42949-021-00022-0**
   > "[paraphrase] Historical HOLC grades are strongly associated with present-day tree canopy across 37 US cities; the design is associational."
   - Context: Structural roots of canopy inequity.

4. **Schwarz et al. (2015), PLOS ONE, doi:10.1371/journal.pone.0122051**
   > "[paraphrase] Urban tree canopy cover is positively associated with income across multiple US cities."
   - Context: Early multi-city evidence that canopy follows wealth.

5. **Benz & Burney (2021), Earth's Future, doi:10.1029/2021EF002016**
   > "[paraphrase] Race and class disparities in surface urban heat extremes are widespread across US urban areas."
   - Context: Heat-extreme disparities, again descriptive.

6. **Sanchez & Reames (2019), Urban Forestry & Urban Greening, doi:10.1016/j.ufug.2019.04.014**
   > "[paraphrase] A socio-spatial analysis of green roofs in Detroit examines whether heat-mitigation infrastructure reaches heat-vulnerable populations."
   - Context: Equity evaluated after deployment, not used to allocate it [details to verify in full text].

**Why This Matters**:
If optimisers maximise total cooling, they may reinforce existing inequity, because the streets with the most pedestrians or the best planting conditions are not necessarily where vulnerable people live. The descriptive literature is strong and consistent (Schwarz et al., 2015; Hsu et al., 2021; McDonald et al., 2021; Locke et al., 2021; Benz & Burney, 2021), so the question is no longer whether disparities exist but how planting decisions should respond to them and at what cost in total benefit. Making equity an explicit objective produces a Pareto front that shows decision-makers the price of fairness at each tree budget, which is information current tools do not provide. Because most evidence is from US cities, applying the framework in Dutch cities also tests whether the efficiency-equity trade-off appears in a different planning and social context.

---

### ✓ Gap 4: Transferability and input-data quality are untested for decision surrogates

**ID**: GAP-004

**Type**: partial

**Significance**: medium

**Feasibility**: Medium. Leave-one-city-out within the Netherlands is straightforward; the Freiburg transfer test depends on Overture height completeness, which is recorded per building and reported.

**Definition**:
Thermal-comfort surrogates are usually trained and evaluated in one city with high-quality local inputs such as LiDAR-derived building and tree heights. Global open inputs now exist, including LiDAR-based LoD2 buildings for the Netherlands, crowd-sourced and partial building heights elsewhere, and satellite-derived 1 m canopy heights with metre-level error. How surrogate accuracy and the resulting planting decisions degrade when moving to another city or to lower-quality height data has not been quantified.

**Evidence** (6 pieces):

1. **Briegel et al. (2023), Urban Climate, doi:10.1016/j.uclim.2022.101359**
   > "[paraphrase] Training data were generated with SOLWEIG for 68 days at hourly resolution in Freiburg, Germany."
   - Context: Single-city training and validation.

2. **Tolan et al. (2024), Remote Sensing of Environment, doi:10.1016/j.rse.2023.113888**
   > "[paraphrase] Global 1 m canopy height maps from RGB imagery reach a mean absolute error of about 2.8 m against aerial lidar."
   - Context: Open canopy input with error large relative to street-tree heights; its effect on Tmrt surrogates is unknown.

3. **Peters et al. (2022), Photogrammetric Engineering & Remote Sensing, doi:10.14358/pers.21-00032r2**
   > "[paraphrase] LoD2 and LoD1 models were reconstructed automatically for all 10 million buildings in the Netherlands from lidar."
   - Context: Provides a high-quality reference against which lower-quality height sources can be compared.

4. **Yu et al. (2024), Urban Climate, doi:10.1016/j.uclim.2024.102003**
   > "[paraphrase] Time-varying canopy density and metre-scale land use were the most important features for street-level temperature estimation."
   - Context: Canopy inputs dominate the prediction, so canopy data error should propagate strongly [transfer setting to verify].

5. **Kamath et al. (2026), Journal of Open Source Software, doi:10.21105/joss.09535**
   > "[paraphrase] SOLWEIG-GPU provides a GPU-accelerated, scriptable SOLWEIG framework intended for urban digital twins."
   - Context: Makes a common physical reference available across cities, enabling controlled transfer tests.

6. **Demuzere et al. (2022), Earth System Science Data, doi:10.5194/essd-14-3835-2022**
   > "[paraphrase] A global 100 m Local Climate Zone map is provided to support urban climate and environmental studies."
   - Context: Enables morphology-stratified patch sampling and transfer analysis by LCZ.

**Why This Matters**:
The promise of open global data is that heat-aware planting tools can be used in cities without LiDAR campaigns, but that promise is only as good as the decisions the tools make with imperfect inputs. A surrogate trained in one city on LiDAR heights (as in Briegel et al., 2023) gives no information about performance where building heights are partial or imputed and canopy heights carry metre-level error (Tolan et al., 2024). Because canopy features dominate street-level predictions (Yu et al., 2024), errors in open canopy data may change which streets are prioritised, not only the predicted temperatures. Quantifying accuracy loss and decision change across cities and data sources tells practitioners when an open-data deployment is trustworthy, and turns a common unstated limitation into a measured result.

---

## Recommendations

✓ **PROCEED**: Gap portfolio meets quality standards.

Next steps:
1. Proceed to originality assessment (Phase 2.2)
2. Discuss core concepts with user (Phase 2.3)
3. Begin outline design (Phase 3)
