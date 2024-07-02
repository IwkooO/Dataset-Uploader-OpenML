from dash import Dash, dcc, html
import dash 
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_ag_grid as dag
from datetime import datetime
import base64
import pandas as pd
import io
from dash.exceptions import PreventUpdate
from langdetect import detect
from functions import publish, chat_api
from feature_detection import return_predictions_of_features
import arff
import csv


#########################
### CSS STYLES
# Styles for the tabs

# tab style by default
tab_style = {
    'border': '1px solid #dee2e6',  
    'padding': '10px',
    'marginRight': '1%',  
    'borderRadius': '8px',  # Rounded corners for all tabs
    'backgroundColor': '#e9ecef',  # Light gray for non-selected tabs
    'textAlign': 'center',
    'color': '#343a40',  # Dark grey text color for better readability
    'cursor': 'pointer',
    'width': '31%',  # Fit 3 tabs on the page
    'boxSizing': 'border-box',
    'whiteSpace': 'nowrap',
    'display': 'inline-block',
    'fontFamily': 'Roboto, sans-serif',
    'fontWeight': 'normal',
    'fontSize': '16px',
    'transition': 'background-color 0.3s',  # Smooth transition for hover and focus
    'userSelect': 'none',  # Disable text selection
}

# tab style when selected
tab_selected_style = {
    'border': '1px solid #007bff',  # Blue border color for selected tab
    'backgroundColor': '#007bff',  # Blue background for selected tab
    'color': 'white',  # White text for selected tab
    'borderRadius': '8px',  # Rounded corners
    'padding': '10px',
    'marginRight': '1%',  # Consistent margin with unselected tabs
    'textAlign': 'center',
    'width': '31%',
    'boxSizing': 'border-box',
    'whiteSpace': 'nowrap',
    'display': 'inline-block',
    'fontFamily': 'Roboto, sans-serif',
    'fontWeight': 'bold',
    'fontSize': '16px',
    'cursor':'pointer',
    'userSelect': 'none',  # Disable text selection
}



# padding for the page content
CONTENT_STYLE = {
    "marginLeft": "15rem",
    "marginRight": "15rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#f7f7f7",
    'borderRadius': '15px',
    "marginTop":'2rem',
    "marginBottom":'2rem',
}


# Main page style

TABS_STYLE = {
    "marginLeft": "10rem",  # Reduced left margin for alignment with content
    "marginRight": "10rem",  # Reduced right margin
    "padding": "1rem 1rem",
    "backgroundColor": "#f7f7f7",  # Consistent background with page
    'borderRadius': '8px',  # Rounded corners
    "marginTop": '1rem',
    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'  # Soft shadow for depth
}

######################### 
### PARAMETERS FOR FUNCTIONS IN LAYOUT AND CALLBACKS

### LANGUAGE NAMES -- Needed for translation between langdetect and accepted API output
language_names = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)"
}

### LICENSE OPTIONS
options_license = [
    {"label": "Public Domain (CC0)", "value": "Public Domain (CC0)"},
    {"label": "Publicly available", "value": "Publicly available"},
    {"label": "Attribution (CC BY)", "value": "Attribution (CC BY)"},
    {"label": "Attribution-ShareAlike (CC BY-SA)", "value": "Attribution-ShareAlike (CC BY-SA)"},
    {"label": "Attribution-NoDerivs (CC BY-ND)", "value": "Attribution-NoDerivs (CC BY-ND)"},
    {"label": "Attribution-NonCommercial (CC BY-NC)", "value": "Attribution-NonCommercial (CC BY-NC)"},
    {"label": "Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)", "value": "Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)"},
    {"label": "Attribution-NonCommercial-NoDerivs (CC BY-NC-ND)", "value": "Attribution-NonCommercial-NoDerivs (CC BY-NC-ND)"},
    {"label": "No License", "value": "No License"}
]





### Intitialize the app

app = Dash(__name__,suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP,'https://use.fontawesome.com/releases/v5.8.1/css/all.css'])
app.server.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024 # 1GB
app.title="OpenML Dataset Uploader"


### Popup for the view dataset head 
popup_modal_view_head=dbc.Modal(
    [
        dbc.ModalHeader("Dataset Head",style={'fontWeight':'bold','textAlign':'center','color':'black','fontSize':'20px'}),
        dbc.ModalBody(
            html.Div(
                [ #Empty dash ag grid that will get populated when dataset is uploaded
                    dag.AgGrid(
                        id='head-grid',
                        columnDefs=[],
                        rowData=[],
                        columnSize='autoSize',
                        dashGridOptions={"pagination": True, "paginationPageSize": 10}
                    )
                ]
        )),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal-dataset-head", className="ml-auto",color='secondary')
        ),
    ],
    id="modal-dataset-head",
    size="xl",
    centered=True,
    backdrop="static",)

### Documentation for Upload Dataset Tab
tab1_content_documentation = dbc.Card(
    dbc.CardBody([
        html.H5("Step 1: Upload Dataset", className="card-title"),
        html.H6("Upload Your File:"),
        html.Ul([
            html.Li("Drag and drop your dataset file into the upload field or click 'Select Files' to browse your computer."),
            html.Li("Accepted formats: JSON, Excel, Parquet, Arff, or CSV. The file size should not exceed 1GB."),
            html.Li("This action will automatically initiate the description generation and language detection processes."),
        ]),
        html.H6("Fill in Dataset Details:"),
        html.Ul([
            html.Li("Name: Enter a name for your dataset in the 'Name' input field. This field is mandatory!"),
            html.Li("Description: After up to one minute, a description will be generated automatically. Verify and edit if necessary."),
        ]),
        html.H6("View Dataset Head:"),
        html.Ul([
            html.Li("Once the dataset is uploaded, use the 'View Dataset Head' button to preview the first few rows and confirm the correct upload."),
        ]),
        html.H6("Proceed to the Next Tab:"),
        html.Ul([
            html.Li("Ensure all fields are correctly filled out. Scroll to the top of the page and select 'Fill Author Data' tab to proceed."),
        ]),
    ])
)
### Documentation for Author Data Tab
tab2_content_documentation = dbc.Card(
    dbc.CardBody([
        html.H5("Step 2: Fill Author Data", className="card-title"),
        html.H6("General Information:"),
        html.Ul([
            html.Li("In this tab, provide general information about the dataset creation. Field include:"),
            html.Ul([
                html.Li("Author"),
                html.Li("Citation"),
                html.Li("Contributor"),
                html.Li("Dataset Creation Date"),
                html.Li("License* (mandatory)"),
            ]),
        ]),
        html.H6("Proceed to the Next Tab:"),
        html.Ul([
            html.Li("After completing the information in this tab, scroll to the top and select 'Edit Feature Data' tab to proceed."),
        ]),
    ])
)
### Documentation for Feature Data tab
tab3_content_documentation = dbc.Card(
    dbc.CardBody([
        html.H5("Step 3: Edit Feature Data", className="card-title"),
        html.H6("Fill in Attribute Information:"),
        html.Ul([
            html.Li("Provide details about the index, target, and ignore attributes in the respective input fields."),
            html.Li("These fields are optional and can be left blank if not applicable."),
        ]),
        html.H6("Generate Feature Types:"),
        html.Ul([
            html.Li("Click the 'Generate Feature Types' button to automatically generate the predicted column features."),
            html.Li("This process may take up to 2 minutes. Wait until the features appear."),
        ]),
        html.H6("Manual Verification and Editing:"),
        html.Ul([
            html.Li("Review the generated features carefully."),
        ]),
        html.H6("Upload the Dataset:"),
        html.Ul([
            html.Li("Once all steps are completed and verified, insert your API key in the provided field below."),
            html.Li("The 'Upload The Dataset' button will now be unlocked. Click it to upload your dataset using the OpenML API."),
            html.Li("Wait up to 3 minutes for the upload to complete. A success message will be displayed upon completion."),
        ]),
    ])
)

### Total documentation modal
documentation_modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("How to Upload Your Dataset",style={'fontSize': '1.25rem', 'fontWeight': 'bold'}),
                dbc.ModalBody(
                    dbc.Tabs(
                        [
                            dbc.Tab(tab1_content_documentation, label="Upload Data"),
                            dbc.Tab(tab2_content_documentation, label="Author Data"),
                            dbc.Tab(tab3_content_documentation, label="Features Data"),
                        ]
                    )
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-modal-documentation", className="ml-auto", style={'backgroundColor': '#343a40', 'color': 'white', 'borderRadius': '5px'})
                ),
            ],
            id="modal_documentation",
            is_open=False,
            size="lg",
        ),
    ]
)


ctx = dash.callback_context
############################################
##### TABS LAYOUT
def get_horizontal_tabs():
    # Page title
    title = html.H2("OpenML Dataset Uploader", className='text-center', style={'fontWeight':'bold','marginBottom': '20px','color': '#343a40'})
    
    tabs = dcc.Tabs(
        id="sidebar-tabs",
        value='tab-home',
        children=[
            dcc.Tab(label='1. Upload Dataset', value='tab-home', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='2. Fill Author Data', value='tab-author', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='3. Edit Feature Data', value='tab-features', style=tab_style, selected_style=tab_selected_style),
        ],
        style={'width': '100%'}  
    )
    # Div with the title and tabs
    return html.Div([title, tabs])

############################################
##### Layout for the Home tab
home_layout = html.Div([
    html.H3('Please Upload Dataset',style={'fontWeight':'bold','textAlign':'center'}),
    html.P('Fill the required boxes',style={'textAlign':'center'}),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',html.A('Select Files', style={'fontWeight': 'bold'}),
                html.Br(),
                html.Span('We accept JSON, Excel, Parquet, Arff or CSV files up to 1GB in size.', style={'fontSize': 'smaller'}), 
            ], className='upload-content'),
            style={
                'width': '100%',  # 100% to make it responsive
                'height': '120px',
                'lineHeight': '60px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '10px',
                'textAlign': 'center',
                'margin': '20px auto',  # Center the upload box
                'padding': '10px',
                'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.1)',  # Add shadow
                'backgroundColor': '#f9f9f9',  # Light background color
                'color': '#333',  # Dark text color for contrast
                'transition': 'all 0.3s',  #  hover effects
                'userSelect': 'none',  # Disable text selection
            },
            className='upload-box',  
            multiple=False  # Allow just 1 file
        ),
    dbc.InputGroup([dbc.InputGroupText("Name*"), dbc.Input(id='Name',placeholder="Fill the name of your dataset",),],className="mb-3"),
    dcc.Loading(id="loading-output", children=[
        dbc.Textarea(id='description-textarea',placeholder="Description",className="mb-3",
                     style={'height': 300})
                     ], type="default"),

    dbc.InputGroup(
        [
            dbc.InputGroupText("Language"),
            dbc.Input(id='language',placeholder="Start 1 upper case , rest lower:, e.g. 'English'.", type="text",
                      ),
        ],
        className="mb-3",
    ),

    ### BUTTONS
     html.Div([
        dbc.Button(
            "View Uploaded Dataset Head",id='view_head_btn',color='secondary',
            className="me-1"),
        ],style={'textAlign': 'center'}  
    ),   

    popup_modal_view_head,

])



############################################
#####  Layout for the Author Dataset tab
author_layout = html.Div([
    html.H3('Edit Author Data',style={'textAlign':'center','fontWeight':'bold'}),
    html.Br(),
    dbc.InputGroup(
        [
        dbc.InputGroupText("Creator"),
        dbc.Input(id='creator',placeholder="The person who created the dataset.", type="text"),
        
        ],className="mb-3"),
        dbc.InputGroup(
            [
                dbc.InputGroupText(html.I(className="fas fa-question-circle", id="citation-target"), style={"cursor": "pointer"}),
                dbc.InputGroupText("Citation"),
                dbc.Textarea(id='citation',
                            style={'height': 150, 'resize': 'none'}),
            ],
            className="mb-3",
        ),
        dbc.Tooltip(
            "Reference(s) that should be cited when building on this data",
            target="citation-target",  # Target is citation field
        ),
    dbc.InputGroup(
        [
            dbc.InputGroupText("Contributor"),
            dbc.Input(id='contributor',placeholder="People who contributed", type="text",
                      ),  
        ],
        className="mb-3",
    ),
    dbc.InputGroup(
        [
            dbc.InputGroupText("Collection Date"),
            dcc.DatePickerSingle(
                    id='collection_date',
                    date='2024-01-01',  # Default/start date
                ),
        ],
        className="mb-3",
    ),

    dbc.InputGroup(
        [
            dbc.InputGroupText("License*", style={'height': 'calc(1.5em + 0.75rem + 2px)'}),
            dcc.Dropdown(
                id='license',
                options=[
                    {"label": "Public Domain (CC0)", "value": "Public Domain (CC0)"},
                    {"label": "Publicly available", "value": "Publicly available"},
                    {"label": "Attribution (CC BY)", "value": "Attribution (CC BY)"},
                    {"label": "Attribution-ShareAlike (CC BY-SA)", "value": "Attribution-ShareAlike (CC BY-SA)"},
                    {"label": "Attribution-NoDerivs (CC BY-ND)", "value": "Attribution-NoDerivs (CC BY-ND)"},
                    {"label": "Attribution-NonCommercial (CC BY-NC)", "value": "Attribution-NonCommercial (CC BY-NC)"},
                    {"label": "Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)", "value": "Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)"},
                    {"label": "Attribution-NonCommercial-NoDerivs (CC BY-NC-ND)", "value": "Attribution-NonCommercial-NoDerivs (CC BY-NC-ND)"}
                ],
                placeholder="Select a license",
                value='',  # Default value or you can set a pre-selected option
                clearable=True,
                style={
                    'height': 'calc(1.5em + 0.75rem + 2px)',  # Match the height of input group text
                    'width': '80%',  # Adjust the width to fit the group
                    'minWidth': '15rem'
                },
            ),
        ],
        className="mb-3",
    ),
])

############################################
##### Layout for the Features Editor tab
features_layout = html.Div([
    html.H3('Features Editor',style={'textAlign':'center','fontWeight':'bold'}),

    ### TARGET ATTRIBUTE
    dbc.InputGroup(
    [
        dbc.InputGroupText(html.I(className="fas fa-question-circle", id="targetattribute-target"), style={"cursor": "pointer"}),
        dbc.InputGroupText("Target Attribute"),
        dbc.Input(id='default_target_attribute',placeholder="Target. Can have multiple (comma-separated). Not a mandatory field", type="text",
                  ),

        dbc.Tooltip(
        " The default target attribute, if it exists. Can have multiple (comma-separated)",
        target="targetattribute-target"),
        
    ],
    className="mb-3"),
    ### IGNORE ATTRIBUTE
    dbc.InputGroup(
        [   
            dbc.InputGroupText(html.I(className="fas fa-question-circle", id="ignoreattribute-target"), style={"cursor": "pointer"}),
            dbc.InputGroupText("Ignore Attribute"),
            dbc.Input(id='ignore_attribute',placeholder="Single Feature or list that should be excluded. Not a mandatory field", type="text",
                      ),
            dbc.Tooltip(
            " Attributes that should be excluded in modelling, such as identifiers and indexes",
            target="ignoreattribute-target"),
        ],
        className="mb-3",
    ),
    ### INDEX COLUMN
    dbc.InputGroup(
        [
            dbc.InputGroupText(html.I(className="fas fa-question-circle", id="indexattribute-target"), style={"cursor": "pointer"}),
            dbc.InputGroupText("Index Column"),
            dbc.Input(id='index_column',placeholder="Name of attribute that is the id. Not a mandatory field", type="text", ###  row_id_attribute=None,
                      ),
            dbc.Tooltip(
            " The attribute that represents the row-id column, if present in the dataset",
            target="indexattribute-target"),
            
        ],
        className="mb-3",
    ),
    # Ag Grid for editing features
    html.Div([
        dcc.Loading(id= 'loading-features',children=[
            dag.AgGrid(
                id='feature-grid',
                columnDefs=[{"headerName": "Feature Name","field":"FeatureName", "sortable": True, "filter": False,"headerClass": "center-header"},
                            {"headerName": "Suggested Feature Type","field":"FeatureType", "sortable": True, "filter": True, 'editable':False,"headerClass": "center-header"},
                            {
                                'headerName': "Publish with Feature Type (EDITABLE)",
                                "headerClass": "center-header",
                                'field': "FeatureTypePublish",
                                'editable': True,  
                                'cellEditor': 'agSelectCellEditor',  
                                'cellEditorParams': {
                                    'values': ['integer', 'floating', 'categorical', 'boolean', 'datetime', 'sentence', 'url', 'embedded-number', 'list', 'not-generalizable', 'context-specific']
                                }
                            }
                    ],
                rowData=[],
                columnSize='responsiveSizeToFit',
                dashGridOptions={"pagination": True, "paginationPageSize": 100})],type="default"),
    ],style={'maxHeight': '80%', 'overflowY': 'auto','width':'90%','margin': '0 auto',}),
    html.Br(),
    html.Div([
        dbc.Button(
            "Generate Feature Types*",id='features_btn',color='secondary',
            className="me-1"),
        ],style={'textAlign': 'center'}  
    ), 



])

############################################
#### Very important to render the content of the tabs in the main layout
# Tab content is now part of the main app layout, created only once.
home_content = html.Div(id='home-content', children=[home_layout], style={'display': 'block'})  # default visible
author_content = html.Div(id='author-content', children=[author_layout], style={'display': 'none'})
features_content = html.Div(id='features-content', children=[features_layout], style={'display': 'none'})

#########################################
## RETURNING CONTENT BASED ON SELECTED TAB

@app.callback(
    [Output('home-content', 'style'),
     Output('author-content', 'style'),
     Output('features-content', 'style')],
    [Input('sidebar-tabs', 'value')]
)
def render_tab_content(tab):
    if tab == 'tab-home':
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
    elif tab == 'tab-author':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
    elif tab == 'tab-features':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}  # Default case

######################################
#### MAIN PAGE LAYOUT
######################################
app.layout = html.Div([
    html.Div(
        [
            get_horizontal_tabs(),
        ],style=TABS_STYLE
    )
    ,html.Br(),
    html.Div([
        
        home_content,
        author_content,
        features_content,
    ], style=CONTENT_STYLE),
    html.Div([
        dbc.Label('After finishing every tab the button below will be enabled.', style={'textAlign': 'center', 'fontWeight': 'bold'}),
        dbc.InputGroup([
            dbc.Input(id='apikey', placeholder="Enter your api key", type="text"),
            dbc.Tooltip(
                "The upload can take a few minutes. Please be patient.",
                target="submit",  # Tooltip target
            ),
            dbc.InputGroupText("API Key"),
        ], className="mb-3",),
        
        dbc.Row([
            dbc.Col(dbc.Button("Upload The Dataset", id="submit", disabled=True, style={'backgroundColor': '#343a40', 'color': 'white', 'borderRadius': '5px'}, className="me-1"), width="auto"),
            dbc.Col(dbc.Button("How to Upload? - Documentation", id="open_documentation", style={'backgroundColor': 'green', 'color': 'white', 'borderRadius': '5px'}, className="me-1"), width="auto"),
        ], justify="between"),
        
        dcc.Loading(html.Div(id='output_upload')),
    ], style=CONTENT_STYLE),

    dcc.Store(id='stored-dataframe'),


    documentation_modal,
])



####################################
### UPLOAD SUBPAGE 1 CALLBACKS
####################################


####################################
### VIEW HEAD CALLBACK
@app.callback(
    Output("modal-dataset-head", "is_open"),
    [Input("view_head_btn", "n_clicks"), Input("close-modal-dataset-head", "n_clicks")],
    [State("modal-dataset-head", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return False

@app.callback(
    [Output('head-grid','rowData'),Output('head-grid','columnDefs')],
    [Input('view_head_btn','n_clicks')],
    [State('stored-dataframe','data')],
    prevent_initial_call=True
)
def view_head(n_clicks,data):
    if data is not None:
        df = pd.read_json(data, orient='split')
        head = df.head()
        columnDefs = [{"headerName": col, "field": col, "sortable": True, "filter": True} for col in head.columns]
        rowData = head.to_dict('records')
        return rowData,columnDefs
    raise PreventUpdate



####################################
### LOAD UPLOADED DATASETS, OUTPUT DESCRIPTION AND LANGUAGE
@app.callback(
    Output('stored-dataframe', 'data'),Output('upload-data','children'),Output('language','value'),Output('description-textarea','value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True,
    
)
def parse_contents(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                ### Get the sample for the sniffer
                sample = io.StringIO(decoded.decode('utf-8')).read(4096)
                
                ### Use sniffer to detect the delimeter
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                has_header = sniffer.has_header(sample)
                
                ### Check if there is a header
                header = 0 if has_header else None
                
                df = pd.read_csv(io.BytesIO(decoded), delimiter=dialect.delimiter, header=header)

                ### Rename the unnamed column to have a name index
                if 'Unnamed: 0' in df.columns:
                    df = df.rename(columns={'Unnamed: 0': 'Index'})
            elif 'arff' in filename:
                # Assume the user uploaded an ARFF file
                decoded = arff.loads(decoded.decode('utf-8'))
                column_names = [attribute_instance[0] for attribute_instance in decoded['attributes']]
                df = pd.DataFrame(decoded['data'],columns=column_names)
                if 'Unnamed: 0' in df.columns:
                    df = df.rename(columns={'Unnamed: 0': 'Index'})
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
                if 'Unnamed: 0' in df.columns:
                    df = df.rename(columns={'Unnamed: 0': 'Index'})
            elif 'json' in filename:
                # Assume that the user uploaded a JSON file
                df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
                if 'Unnamed: 0' in df.columns:
                    df = df.rename(columns={'Unnamed: 0': 'Index'})
            elif 'parquet' in filename:
                # Assume the user uploaded a Parquet file
                df = pd.read_parquet(io.BytesIO(decoded))
                if 'Unnamed: 0' in df.columns:
                    df = df.rename(columns={'Unnamed: 0': 'Index'})
            ### Need to make arfff
            else:
                return pd.DataFrame().to_json(date_format='iso', orient='split'),html.Div("Not accepted file type"),'',''
        except Exception as e:
            df=pd.DataFrame()
            return df.to_json(date_format='iso', orient='split'),html.Div(f"Dataset upload has failed. {e}"),'',''
        
        sampled_values = {col: df[col].sample(5).tolist() for col in df.columns}
        sampled_values_str = str(f"Dataset name: {filename}, Columns : ")+", ".join([f"{col}: {values}" for col, values in sampled_values.items()])
        

        language_prediciton = detect(sampled_values_str)
        language = language_names[language_prediciton]
        return df.to_json(date_format='iso', orient='split'),html.Div(f"{filename} has been uploaded successfully."),language, chat_api(sampled_values_str)
    raise PreventUpdate


    

####################################
### FEATURES SUBPAGE 3 CALLBACKS
####################################


####################################
## RETURN FEATURES ON BUTTON CLICK
@app.callback(
    Output('feature-grid','rowData'),
    [Input('features_btn', 'n_clicks')],
    [State('stored-dataframe','data')]
)
def load_features(n, stored_data):
    ctx = dash.callback_context
    if ctx.triggered_id == "features_btn":
        if stored_data !=None:
            # Load the dataframe from JSON
            df = pd.read_json(stored_data, orient='split')
            feature_predictions = return_predictions_of_features(df)
            #feature_types = feature_predictions[2]
            feature_proposal = feature_predictions[0]

            # Shown data in the table
            rowData = [
                {"FeatureName": df.columns[col], "FeatureType":str(feature_proposal[col]),"FeatureTypePublish":str(feature_proposal[col])} 
                for col in range(len(df.columns))]
            return rowData
    return []


######################################
### SUBMIT

### PUBLISH DATASET
@app.callback(
    Output('output_upload','children'),
    Input('submit','n_clicks'),
    [State('stored-dataframe', 'data'),State('language','value'),State('Name', 'value'),State('description-textarea','value'),
     State('license','value'),State('creator','value'),State('contributor','value'),State('citation','value'),
     State('collection_date','date'),State('feature-grid','rowData'),State('default_target_attribute','value'),State('ignore_attribute','value'),
     State('index_column','value'),State('apikey','value')],
    prevent_initial_call = True

)
def publish_dataset(n,data,lang,name,desc,license,creator,contributor,citation,date,features,target,ign,indx,api):
    ctx = dash.callback_context
    if ctx.triggered_id == "submit":

        if api is None or api == '':
            return "Please enter your API key"
        date = datetime.strptime(date, '%Y-%m-%d')
        date = date.strftime('%d-%m-%Y')

        lang = str(lang)
        name = str(name)
        desc = str(desc)

        name = name.replace(' ','_')
        name = name.replace('“','"')
        desc = desc.replace('“','"')
        desc=desc.replace('”','"')
        desc = desc.replace('‘',"'")
        desc = desc.replace('’',"'")
        desc = desc.replace('—','-')
        desc = desc.replace('–','-')
        desc = desc.replace('…','...')
        
        license = str(license)
        creator = str(creator)
        contributor = str(contributor)
        citation = str(citation)

        if features is None:
            return "No features have been detected / check dataset upload"
        if data is None:
            return "Upload was unsuccesfull , check if format of file matches"
        try:
            df = pd.read_json(data, orient='split')
        except:
            return "There was a problem with reading the dataset, check if dataset is alright"

        choice_to_arff = []
        #### TRANSLATE CHOICES MADE BY USER TO APPROPRIATE ARFF TYPES
        for i in features:
            if i['FeatureTypePublish'] == 'not-generalizable':
                choice_to_arff.append(('STRING'))
            elif i['FeatureTypePublish'] == 'context-specific':
                choice_to_arff.append(('STRING'))
            elif i['FeatureTypePublish'] == 'embedded-number':
                choice_to_arff.append(('STRING'))
            elif i['FeatureTypePublish'] == 'list':
                choice_to_arff.append(('STRING'))
            elif i['FeatureTypePublish'] == 'url':
                choice_to_arff.append(('STRING'))
            elif i['FeatureTypePublish'] == 'sentence':
                choice_to_arff.append(('STRING'))
            elif i['FeatureTypePublish'] == 'datetime':
                choice_to_arff.append(('STRING'))
            elif i['FeatureTypePublish'] == 'boolean':
                choice_to_arff.append([str(value) for value in df[i['FeatureName']].unique()])
            elif i['FeatureTypePublish'] == 'categorical':
                choice_to_arff.append([str(value) for value in df[i['FeatureName']].unique()])
            elif i['FeatureTypePublish'] == 'floating':
                choice_to_arff.append(('REAL'))
            elif i['FeatureTypePublish'] == 'integer':
                choice_to_arff.append(('INTEGER'))
            else:
                choice_to_arff.append(('STRING'))

        features = [(features[row_id]['FeatureName'], choice_to_arff[row_id]) for row_id in range(len(features))]


        try:
            features  = [tuple(item) for item in features]
        except:
            return "No features have been detected / check dataset upload"
        
        ## Check if any of the inputs is None
        if target =='None' or target == '':
            target = None
        if ign =='None' or ign == '':
            ign = None
        if indx =='None' or indx == '':
            indx = None

        ### Try to upload the dataset with predicted features
        try:
            return publish(data=df,name=name,description=desc,license=license,creator=creator,contributor=contributor,collection_date=date,language=lang,attributes=features,default_target_attribute=target,ignore_attribute=ign,citation=citation,api=api,row_id=indx) + "     ||| Method:  Features have been choosen correctly" 
        except Exception as e:
            ### Try to upload the dataset with auto detection of features
            try: 
                return publish(data=df,name=name,description=desc,license=license,creator=creator,contributor=contributor,collection_date=date,language=lang,attributes='auto',default_target_attribute=target,ignore_attribute=ign,citation=citation,api=api,row_id=indx) + "     ||| Method: Features have been choosen automatically"
            except Exception as a:
                ### Show the errors that occured
                return f"Error: {a} and {e}"


####################################
### UNLOCK BUTTON AFTER FEATURES

@app.callback(
    Output('submit', 'disabled'),
    Input('feature-grid', 'rowData')
)
def update_button_status(data):
    return len(data) == 0

####################################
### DOCUMENTATION MODAL
@app.callback(
    Output("modal_documentation", "is_open"),
    [Input("open_documentation", "n_clicks"), Input("close-modal-documentation", "n_clicks")],
    [State("modal_documentation", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return False




##################################
### START APP
##############################
if __name__ == '__main__':
    app.run_server(debug=True)
