import pandas as pd
import regex as re


def parse_single_file (input_file, output_file):
    df = pd.read_csv(f'{input_file}')


    df.output = df.output.apply(lambda x:x.split('Output')[-1])
    df['pattern'] = df['05']+'|'+df['01']+'|'+df['02']


    def parse_output(a,b):
        try:
            output = re.search(a,b).group()
        except: output = None

        return output

    df['parsed_output'] = df.apply(lambda x:parse_output(x.pattern,x.output),axis=1)
    df = df.dropna().drop_duplicates(subset=['id','05','01','02'])

    l = list()

    for _,item in df.iterrows():
        if item.parsed_output==item['05']:
            l.append('ann05')
        elif item.parsed_output==item['01']:
            l.append('ann01')
        else:
            l.append('ann02')

    df['label'] = l
    df.to_csv(f'{output_file}',index=False)





