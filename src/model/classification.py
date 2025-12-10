import pandas as pd
import random
import csv
import transformers
import torch
from tqdm import tqdm



def create_prompt(row,seed):
  
  text = row["tweet"]
  opt_ann05 = row['cleaned_cl_ann05']
  opt_ann01 = row['cleaned_cl_ann01']
  opt_ann02 = row['cleaned_cl_ann02']
  random.seed(seed)
  list_options = [opt_ann05, opt_ann01, opt_ann02]
  random.shuffle(list_options)

  instruction = {"prelude": "Ti viene fornita in input (Input) una frase estratta dai social media, insieme a tre possibili stereotipi (Opzioni).",
        "task": "Il tuo compito è individuare quale stereotipo è implicito nella frase, scegliendo tra le opzioni fornite." ,
        "instr": "Restituisci in output (Output) una singola opzione, sotto forma di lista Python (es. ['Opzione 1']).",
        "input": f"Input: {text}",
        "options": f"Opzioni: {list_options}",
        "output": "Output:"}

  prompt = f"{instruction['prelude']} {instruction['task']} {instruction['instr']}\n {instruction['input']}\n {instruction['options']}\n {instruction['output']}"

  return prompt



def classify(df, pred_path, processed_fileame, prediction_filename, seed=42):
    file = open(f'{pred_path}/{prediction_filename}.csv',mode='w')

    writer = csv.DictWriter(file,fieldnames=['id','span','output',"05","01","02"])
    writer.writeheader()


    df = df[df["cluster_5_nome_ann02"] != 'None/Doubt']
    df = df[df["cluster_5_nome_ann02"] != 'None/Doubt']

    dict_ann05 =  {'SonoSfruttatori': "Sono degli sfruttatori",
               'SonoMinaccia': "Sono una minaccia",
               'RovinanoItalia': "Rovinano l'Italia",
               'SonoTutelati': "Sono tutelati",
               'SonoEstremistiReligiosi': "Sono degli estremisti religiosi"}

    dict_ann01 =  {'SonoParassiti': "Sono dei parassiti",
               'SonoSubdoli': "Sono subdoli",
               'SonoImmorali': "Sono immorali",
               'SonoIncompatibiliConNoi': "Sono incompatibili con noi",
               'SonoProblema': "Sono un problema"}

    dict_ann02 = {'FannoQuelloCheVoglionoSenzaContribuire':  "Fanno quello che vogliono senza contribuire",
            'SonoPericolosi': "Sono pericolosi",
            'PeggioranoLeNostreCondizioniDiVita': "Peggiorano le nostre condizioni di vita",
            'HannoCulturaDiversaDallaNostra': "Hanno una cultura diversa dalla nostra",
            'PortanoDegrado': "Portano degrado"}

    df["cleaned_cl_ann05"] = df["cluster_5_nome_ann05"].map(dict_ann05)
    df["cleaned_cl_ann01"] = df["cluster_5_nome_ann01"].map(dict_ann01)
    df["cleaned_cl_ann02"] = df["cluster_5_nome_ann02"].map(dict_ann02)


    df = df [['id', 'annotatore', 'tweet', 'chunk', 'annotazione','annotazioni_parsate',
          'cleaned_cl_ann05', 'cleaned_cl_ann01', 'cleaned_cl_ann02']]

    dataset = df
    print(len(dataset))

    dataset["prompt"] = dataset.apply(lambda row: create_prompt(row, seed), axis=1)

    dataset.to_csv(f'{pred_path}/{processed_fileame}.csv',index=False)



    model_id = "sapienzanlp/Minerva-7B-instruct-v1.0"  # Replace with a compatible model ID



    transformers.set_seed(seed)

    # Initialize the pipeline.
    pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.bfloat16},
)


    for _,item in tqdm(dataset.iterrows(),total=len(dataset)):
        output = pipeline(
        item.prompt,
        max_new_tokens=15,
    )


    writer.writerow({'id':item.id,'span':item.chunk,'output':output,'05':item.cleaned_cl_ann05,'01':item.cleaned_cl_ann01,'02':item.cleaned_cl_ann02})

    file.close()