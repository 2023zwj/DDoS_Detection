import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--DecisionNum', type=int, default=100, help='The num of the decision trees')
parse.add_argument('--ratio', type=float, default=1, help='The ratio of the samples')
parse.add_argument('--featureNum', type=int, default=5, help='The num of the feature')
parse.add_argument('--MaxDepth',type=int, default=20, help='The max depth of the decision tree')
parse.add_argument('--MinSample',type=int,default=25, help='The min count of the samples')
