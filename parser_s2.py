import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--path', default='data/dataset.pkl', type=str, help="data path")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--batch', default=1024, type=int, help='batch size')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--num_neg_samples', default=127, type=int, help='num_neg_samples')
    parser.add_argument('--gnn_layer', default=1, type=int, help='number of gnn layers')
    parser.add_argument('--dropout', default=0.2, type=float, help='rate for edge dropout')
    parser.add_argument('--lambda1', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--temp', default=.25, type=float, help='temperature in cl loss')
    parser.add_argument('--activation', default=0.1, type=float, help='LeakyReLU Negative Slope')
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')
    return parser.parse_args()
args = parse_args()

# batch_size, batch_type, expand_factor, num_workers=8, shuffle=True, pos_dim=128