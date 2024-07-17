import argparse

parse = argparse.ArgumentParser() 
parse.add_argument('--MaxDepth',type=int, default=20, help='The max depth of the decision tree')
parse.add_argument('--MinSample',type=int,default=25, help='The min count of the samples')

