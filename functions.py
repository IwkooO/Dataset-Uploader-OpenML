import pandas as pd
import openml as oml
import dash_ag_grid as dag
from openai import OpenAI


###################################
## Function to publish the dataset to OpenML given all the variables

def publish(data,name,description,license,creator,contributor,collection_date,language,attributes,default_target_attribute,ignore_attribute,citation,row_id,api):
    oml.config.apikey = api


    my_data = oml.datasets.functions.create_dataset(
        name=name, description=description,
        licence=license, data=data, creator=creator, contributor=contributor, collection_date=collection_date,
        language=language, attributes=attributes, default_target_attribute=default_target_attribute, ignore_attribute=ignore_attribute,
        citation=citation,row_id_attribute=row_id
    )

    # Share the dataset on OpenML
    try:
        my_data.publish()
        return (f"URL for dataset: {my_data.openml_url}")
    except Exception as e:
        # Catching the general exception and returning its message
        return f'An error occurred: {str(e)}'


###################################
## GPT API call

def chat_api(sampled_values_str):
    ### change to os 
    client = OpenAI(
    # This is the default and can be omitted
    api_key="",
    )
    # Persona pattern prompt
    # Other prompts can be also explored, just change the text_persona variable

    text_persona = """You are the creator of a dataset. You want to upload the dataset to an online repository.
    You are requested to provide a dataset description.
    Knowing the column names and their sample values you will write a concise
    and informative description within 250 words limit without use only ASCII standard characters.
    A template for this task is as follows:
    Description:
    Atttirbute Description:
    Use Case:

    """
    total_prompt = text_persona+sampled_values_str # Concatenating the prompt and the sampled values

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": total_prompt,
            }
        ],
        #model="gpt-3.5-turbo",)
        model = "gpt-4-0125-preview",)
    return chat_completion.choices[0].message.content


