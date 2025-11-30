from models.gnn import GNN
from models.topo_gnn import TopoGNN


def get_model(args):
    if args.gnn in ["gcn", "gin", "gps", "pna"]:
        if args.diagram_type in ["spectre", "rephine", "standard", "fast_spectre"]:
            model = TopoGNN(
                hidden_dim=args.hidden_dim,
                depth=args.depth,
                num_node_features=args.num_node_features,
                num_classes=args.num_classes,
                gnn=args.gnn,
                num_filtrations=args.num_filtrations,
                filtration_hidden=args.filtration_hidden,
                out_ph_dim=args.out_dim,
                positional_encoding_dim=args.positional_encoding_dim,
                diagram_type=args.diagram_type,
                ph_pooling_type=args.ph_pooling_type,
                dim1=args.dim1,
                sig_filtrations=args.sig_filtrations,
                global_pooling=args.global_pooling,
                batch_norm=args.bn,
               out_dim_eigen_deep_set=args.out_dim_eigen_deep_set,
                deg=args.deg
            )
        else:
            model = GNN(
                hidden_dim=args.hidden_dim,
                depth=args.depth,
                num_node_features=args.num_node_features,
                num_classes=args.num_classes,
                gnn=args.gnn,
                deg=args.deg,
                global_pooling=args.global_pooling,
                batch_norm=args.bn
            )
    else:
        print("I dont know what to do with this combination of diagrams and GNN")

    return model
