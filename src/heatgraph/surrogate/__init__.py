"""Block E: heterogeneous GNN surrogate of hourly segment UTCI under tree edits.

  graphs.py  SceneGraph: planting-independent precomputation + HeteroData per (planting, hour)
  model.py   HeatGraphGNN: GATv2 slot->segment (edge features) + SAGE segment<->segment
  train.py   fit / predict / evaluate (CPU now; GPU only when explicitly requested later)

Counterfactual training: labels for random plantings come from the Block C engine
(`heatgraph.synthetic.generate_labels` on toy scenes; the same engine on real patches).
"""
