import pandas as pd
import argparse
import numpy as np
import sys

pd.set_option('display.max_rows', None)



def load_latex_table(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    lines = [line.strip().rstrip('\\\\') for line in lines]
    data = [line.split('&') for line in lines]
 #   print(data[0])
    if data[0][0] != 'type':
        headers = 'type&dim&batchsize&ppx&pos&neg&split&frozen&acc&prec&rec&f1'.split('&')
        rows = data
    else:
        headers = data[0]
#    new_line = 'type&dim&batchsize&ppx&pos&neg&split&frozen&acc&prec&rec&f1\\'
        rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    for col in ['acc', 'prec', 'rec', 'f1']:
        df[col] = df[col].astype(float)
    grouped_mean = df.groupby(['type', 'dim', 'batchsize', 'ppx', 'pos', 'neg', 'split', 'frozen']).mean()
    grouped_size = df.groupby(['type', 'dim', 'batchsize', 'ppx', 'pos', 'neg', 'split', 'frozen']).size()
    grouped_mean['size'] = grouped_size
    return grouped_mean

def latexline(df, f=0, s=999):
    means_float = [np.round(np.mean(df[x]),3) for x in ['acc', 'prec', 'rec', 'f1']]
    means = [str(np.round(np.mean(df[x]),3))[1:] for x in ['acc', 'prec', 'rec', 'f1']]
    #means = [str(np.round(np.mean(df[x]),3))[1:] for x in ['acc', 'prec', 'rec', 'f1']]
    stds = [str(np.round(np.std(df[x]),3))[1:] for x in ['acc', 'prec', 'rec', 'f1']]
    print("&${}\pm{}$&${}\pm{}$&${}\pm{}$&${}\pm{}$\\\\".format(means[0], stds[0],means[1], stds[1],means[2], stds[2],means[3], stds[3]))
    return means_float
    #return [np.round(np.mean(df[x]),3)[1:] for x in ['acc', 'prec', 'rec', 'f1']]
    #latexline(grouped_df, ['eq', '210x700', '64', '120', '240', '750'])
    #latexline(grouped_df, ['eq', '224x224', '16', '---', '---', '---'])
    
def load_latex_table_from_zeroshot(args):
    filename = args.table
    ### Read in val
    df = pd.read_csv(filename)
    df['window_dim'] = df['window_dim'].astype(str)
    if not args.window_dim[0] == 0:
        df = df.loc[df['window_dim'] == '[{}, {}]'.format(args.window_dim[0], args.window_dim[1])]
    ## Decide on what basis you want to select
    df['score'] = df[args.metric].sum(axis=1) #+ df['f1'] + df['prec']
    #print(df)
    #sys.exit(0)
    df = df.loc[df.groupby(['seed'])['score'].idxmax()].reset_index(drop=True)
    df['threshold'] = df['threshold'].astype(float)
    test = pd.read_csv(filename.replace('val', 'test'))
    for col in df.columns:
        if col != 'score':
            if df[col].dtype != test[col].dtype:
                test[col] = test[col].astype(df[col].dtype)
    test = test.drop(columns = ['latexline'])

    results = pd.DataFrame(columns = test.columns)
    for col in test.columns:
        if test[col].dtype != results[col].dtype:
            results[col] = results[col].astype(test[col].dtype)
    for i in range(len(df)):
        
        window_dim = df.iloc[i]['window_dim']
        if not args.threshold:
            threshold = df.iloc[i]['threshold']
        else:
            threshold = args.threshold
        seed = df.iloc[i]['seed']
        #print(window_dim, threshold, seed)
        #print(test.loc[(test['window_dim'] == window_dim) & (test['threshold'] == threshold)& (test['seed'] == seed)])
        try:
            selected_row_from_test = test.loc[(test['window_dim'] == window_dim) & (test['threshold'] == threshold) & (test['seed'] == seed)].reset_index(drop=True).iloc[0]
        except IndexError:
            print(window_dim, threshold, seed)
            print(test.loc[(test['window_dim'] == window_dim) & (test['threshold'] == threshold)& (test['seed'] == seed)])
        results.loc[len(results)] = selected_row_from_test
    print(results)
    _ = latexline(results)

    #print(test)
    #then find test scores based on these val values

def main(args):
        #df = load_latex_table('latex2_ATM_buildings_square_balanced_tests.txt')
        #df = load_latex_table('latex2_ATM_buildings_square_tests.txt')
        if 'zeroshot' in args.table:
            df = load_latex_table_from_zeroshot(args)
        else:
            df = load_latex_table(args.table)

        print(df)

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--table', type=str, default='latex_ATM_df_square_balanced_tests.txt')
        parser.add_argument('--metric', nargs='+', type=str, default=['acc'])
        parser.add_argument('--window_dim', nargs='+', type=int, default=[0,0])
        parser.add_argument('--threshold', type=float, default=0)
        args = parser.parse_args()
        main(args)