import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', type=int, default=1000)
parse.add_argument('--epochs', type=int, default=20)
parse.add_argument('--feature_dis', type=int, default=25)
parse.add_argument('--lr', type=float, default=0.001)
