1

# # select multiple files and chat with one or multiple # #

# import os
# import streamlit as st
# import pandas as pd
# from docx import Document
# from docx.shared import Pt
# from langchain_community.llms import openai
# from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
# from langchain_openai import ChatOpenAI
# from langchain.agents.agent_types import AgentType
# from streamlit_chat import message
# from dotenv import load_dotenv
# import zipfile

# load_dotenv()
# openai.api_key = os.getenv('OPENAI_API_KEY')
# st.set_option('client.showErrorDetails', False)
# st.set_page_config(page_title="CHAT BOT", page_icon=":robot_face:")
# st.markdown("<h1 style='text-align: center;'>DATA ANALYST</h1>", unsafe_allow_html=True)

# # Add the dropdown widget for user selection
# option = st.selectbox("**Select Option:**", ("Chat with files", "Chat with Database"))

# if option == "Chat with files":
#     if 'generated' not in st.session_state:
#         st.session_state['generated'] = []
#     if 'past' not in st.session_state:
#         st.session_state['past'] = []
#     if 'messages' not in st.session_state:
#         st.session_state['messages'] = [{"role": "system", "content": "You are a helpful assistant."}]

#     uploaded_files = st.file_uploader("**Choose file(s):**", accept_multiple_files=True)

#     # Process the uploaded file(s)
#     dataframes = []
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             file_extension = os.path.splitext(uploaded_file.name)[1]
#             if file_extension.lower() == ".zip":
#                 # Create a temporary directory to extract the contents of the .zip file
#                 temp_dir = "temp_zip_extract"
#                 os.makedirs(temp_dir, exist_ok=True)
                
#                 # Extract the contents of the .zip file to the temporary directory
#                 with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
#                     zip_ref.extractall(temp_dir)
                
#                 # Process each extracted file
#                 for filename in os.listdir(temp_dir):
#                     filepath = os.path.join(temp_dir, filename)
#                     if filename.endswith(".csv"):
#                         dataframe = pd.read_csv(filepath)
#                     elif filename.endswith((".xls", ".xlsx", "xlsm", ".xlsb")):
#                         dataframe = pd.read_excel(filepath)
#                     elif filename.endswith(".txt"):
#                         # Check the second line of the file to determine the delimiter
#                         with open(filepath, 'r') as f:
#                             # Skip the first line
#                             next(f)
#                             second_line = f.readline().strip()
#                             delimiter = "|" if "|" in second_line else "\t"  # Assume "|" delimiter for .psv, "\t" for .tsv

#                             # Reset file pointer to the beginning of the file
#                             f.seek(0)

#                             # Read the file using pandas read_csv
#                             dataframe = pd.read_csv(f, delimiter=delimiter)
#                     else:
#                         st.warning(f"Unsupported file format: {filename}")
#                         continue
#                     dataframes.append(dataframe)
                
#                 # Remove the temporary directory and its contents
#                 if os.path.exists(temp_dir):
#                     for root, dirs, files in os.walk(temp_dir, topdown=False):
#                         for name in files:
#                             os.remove(os.path.join(root, name))
#                         for name in dirs:
#                             os.rmdir(os.path.join(root, name))
#                     os.rmdir(temp_dir)
                
#             elif file_extension.lower() in [".csv", ".xls", ".xlsx", "xlsm", ".xlsb", ".txt"]:
#                 # Process single file uploads as usual
#                 if file_extension.lower() == ".csv":
#                     dataframe = pd.read_csv(uploaded_file)
#                 elif file_extension.lower() in [".xls", ".xlsx", "xlsm", ".xlsb"]:
#                     dataframe = pd.read_excel(uploaded_file)
#                 elif file_extension.lower() == ".txt":
#                     # Check the second line of the file to determine the delimiter
#                     with uploaded_file as f:
#                         # Skip the first line
#                         next(f)
#                         second_line = f.readline().decode().strip()  # decode bytes to string
#                         delimiter = "|" if "|" in second_line else "\t"  # Assume "|" delimiter for .psv, "\t" for .tsv

#                         # Reset file pointer to the beginning of the file
#                         f.seek(0)

#                         # Read the file using pandas read_csv
#                         dataframe = pd.read_csv(f, delimiter=delimiter)
#                 dataframes.append(dataframe)
#             else:
#                 st.warning(f"Unsupported file format: {file_extension}")

#     if dataframes:
#         for idx, dataframe in enumerate(dataframes):
#             st.write(f"### File {idx+1}:")
#             st.write(dataframe)

#     if dataframes:
#         selected_file_idxs = st.multiselect("**Select files for chatting:**", options=range(1, len(dataframes) + 1))
#         selected_dataframes = [dataframes[idx - 1] for idx in selected_file_idxs]  # Get the selected dataframes
#         if selected_file_idxs:
#             st.markdown("<h3 style='text-align: center;'>CHAT WITH BOT</h3>", unsafe_allow_html=True)

#             def generate_response(input_text, selected_dataframes):
#                 response = ""
                
#                 if len(selected_dataframes) == 1:
#                     if "summary" in input_text.lower():
#                         summary_data = round(selected_dataframes[0].describe(include="all"), 2).T.reset_index()
#                         response += "**Summary of Selected File:**\n"
#                         response += summary_data.to_string() + "\n"
#                     elif "missing values" in input_text.lower():
#                         missing_values = selected_dataframes[0].isnull().sum()
#                         response += "**Missing values in Selected File:**\n"
#                         response += missing_values[missing_values > 0].to_string() + "\n"
                    # elif "correlation" in input_text.lower():
                    #     # Calculate correlation matrix for numeric columns
                    #     correlation_matrix = dataframe.corr()
                    #     response = "Here's the correlation matrix:\n"
                    #     response += correlation_matrix.to_string()

                    #     # Plot correlation heatmap
                    #     plt.figure(figsize=(10, 8))
                    #     sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
                    #     plt.title("Correlation Heatmap")
                    #     plt.xlabel("Features")
                    #     plt.ylabel("Features")
                    #     st.pyplot()

#                     else:
#                         agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
#                                                               selected_dataframes[0], verbose=True,
#                                                               agent_type=AgentType.OPENAI_FUNCTIONS,
#                                                               agent_executor_kwargs={"handle_parsing_errors": True})
#                         response += "**Response for Selected File:**\n"
#                         response += agent.run(input_text) + "\n"
#                 elif len(selected_dataframes) > 1:
#                     agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
#                                                           selected_dataframes, verbose=True,
#                                                           agent_executor_kwargs={"handle_parsing_errors": True})
#                     response += "**Response:**\n"
#                     response += agent.run(input_text) + "\n"
#                 return response

#             response_container = st.container()
#             download_button_container = st.container()
#             input_container = st.container()

#             with input_container:
#                 with st.form(key='my_form', clear_on_submit=True):
#                     user_input = st.text_area("**You:**", key='input', height=st.session_state.get('input_height', 50))
#                     st.session_state['input_height'] = len(user_input.split('\n')) * 20
#                     submit_button = st.form_submit_button(label='Send')

#                 if submit_button and user_input:
#                     try:
#                         response = generate_response(user_input, selected_dataframes)
#                         st.session_state['past'].append(user_input)
#                         st.session_state['generated'].append(response)
#                     except Exception as e:
#                         st.error("An error occurred: {}".format(e))

#             if st.session_state['generated']:
#                 with response_container:
#                     st.markdown("---")
#                     for i, (user_input, generated_response) in enumerate(zip(st.session_state["past"], st.session_state["generated"])):
#                         message(user_input, is_user=True, key=str(i) + '_user')
#                         message(generated_response, key=str(i))

#                 with download_button_container:
#                     download_button_clicked = False
#                     download_button = st.button("**Download Chat History**")

#                     if download_button and not download_button_clicked:
#                         download_button_clicked = True
#                         chat_history_file_path = "chat_history.docx"
#                         document = Document()
#                         document.add_heading('Chat History', level=1)
#                         document.add_paragraph()

#                         for user_input, generated_response in zip(st.session_state["past"], st.session_state["generated"]):
#                             user_paragraph = document.add_paragraph()
#                             user_paragraph.add_run("You: ").bold = True
#                             user_paragraph.add_run(user_input).font.size = Pt(12)

#                             bot_paragraph = document.add_paragraph()
#                             bot_paragraph.add_run("Bot: ").bold = True
#                             bot_paragraph.add_run(generated_response).font.size = Pt(12)
#                             document.add_paragraph()

#                         document.save(chat_history_file_path)
#                         st.markdown(f"[Download Chat History](sandbox:/path/{chat_history_file_path})", unsafe_allow_html=True)
# else:
#     # Add functionality for chatting with a database
#     st.write("This functionality is not yet implemented.")


2

# # some useful code # #
                # elif len(dataframes) > 1:
                #     if any(keyword in input_text.lower() for keyword in ["compare", "difference"]):
                #         response += "**Comparison of Selected Files by Columns and DataTypes Wise:**\n"
                #         for i in range(len(dataframes)):
                #             for j in range(i+1, len(dataframes)):
                #                 file1, file2 = dataframes[i], dataframes[j]
                #                 response += f"\n**Comparison between Selected File {i+1} and File {j+1}:**\n\n"
                #                 columns_info_file1 = {col: file1[col].dtype for col in file1.columns}
                #                 columns_info_file2 = {col: file2[col].dtype for col in file2.columns}
                #                 if set(file1.columns) == set(file2.columns):
                #                     response += "*Both files have the same columns.*"
                #                     if columns_info_file1 == columns_info_file2:
                #                         response += "*And their data types are the same.*\n\n"
                #                         response += "\n".join([f"Column '{col}' ({dtype})" for col, dtype in columns_info_file1.items()])
                #                     else:
                #                         response += "*But their data types are different.*\n\n"
                #                         response += f"Columns and data types in File 1:\n{columns_info_file1}\n\n"
                #                         response += f"Columns and data types in File 2:\n{columns_info_file2}\n\n"
                #                 else:
                #                     response += "*Both files have the different columns.*\n"
                #                     response += f"File 1 has columns: {', '.join(file1.columns)}\n\n"
                #                     response += f"File 2 has columns: {', '.join(file2.columns)}\n"
                #     else:
                #         agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
                #                                               dataframes[0], verbose=True,
                #                                               agent_type=AgentType.OPENAI_FUNCTIONS,
                #                                               agent_executor_kwargs={"handle_parsing_errors": True})
                #         response += "**Response:**\n"
                #         response += agent.run(input_text) + "\n"


3

# # Ask user to enter OpenAI API key # #
    
# openai_api_key = st.text_input("**Enter your OpenAI API Key:**", type='password', help="https://platform.openai.com/api-keys")

# if openai_api_key == "":
#     st.info("Please add your OpenAI API key to continue.")
#     st.stop()

# # Create a button for the user to submit their API key
# if st.button('Submit'):
#     # Set the OpenAI API key as an environment variable
#     os.environ["OPENAI_API_KEY"] = openai_api_key
#     # Set the OpenAI API key directly
#     OpenAI.api_key = openai_api_key

#     # Check if the API key is valid by making a simple API call
#     try:
#         client = OpenAI()
#         checking = client.generate(["hi chatgpt"])
#     except Exception as e:
#         st.error("Error testing API key: {}".format(e))
#     else:
#         st.success("API key is valid!")
    

4

# # Select multiple files process at a time all # #

# import os
# import streamlit as st
# import pandas as pd
# from docx import Document
# from docx.shared import Pt
# from langchain_community.llms import openai
# from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
# from langchain_openai import ChatOpenAI
# from langchain.agents.agent_types import AgentType
# from streamlit_chat import message
# from dotenv import load_dotenv
# import zipfile

# load_dotenv()

# # Loading API Key
# openai.api_key = os.getenv('OPENAI_API_KEY')

# # Hide traceback
# st.set_option('client.showErrorDetails', False)

# # Set page title and header
# st.set_page_config(page_title="CHAT BOT", page_icon=":robot_face:")
# st.markdown("<h1 style='text-align: center;'>DATA ANALYST</h1>", unsafe_allow_html=True)

# # Initialise session state variables
# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []
# if 'past' not in st.session_state:
#     st.session_state['past'] = []
# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [{"role": "system", "content": "You are a helpful assistant."}]

# # Allow user to upload file(s)
# uploaded_files = st.file_uploader("**Choose file(s):**", accept_multiple_files=True)

# # Process the uploaded file(s)
# dataframes = []
# uploaded_file_names = []
# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         file_extension = os.path.splitext(uploaded_file.name)[1]
#         if file_extension.lower() == ".zip":
#             # Create a temporary directory to extract the contents of the .zip file
#             temp_dir = "temp_zip_extract"
#             os.makedirs(temp_dir, exist_ok=True)
            
#             # Extract the contents of the .zip file to the temporary directory
#             with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
#                 zip_ref.extractall(temp_dir)
            
#             # Process each extracted file
#             for filename in os.listdir(temp_dir):
#                 filepath = os.path.join(temp_dir, filename)
#                 if filename.endswith(".csv"):
#                     dataframe = pd.read_csv(filepath)
#                 elif filename.endswith((".xls", ".xlsx", "xlsm", ".xlsb")):
#                     dataframe = pd.read_excel(filepath)
#                 elif filename.endswith(".txt"):
#                     # Check the second line of the file to determine the delimiter
#                     with open(filepath, 'r') as f:
#                         # Skip the first line
#                         next(f)
#                         second_line = f.readline().strip()
#                         delimiter = "|" if "|" in second_line else "\t"  # Assume "|" delimiter for .psv, "\t" for .tsv

#                         # Reset file pointer to the beginning of the file
#                         f.seek(0)

#                         # Read the file using pandas read_csv
#                         dataframe = pd.read_csv(f, delimiter=delimiter)
#                 else:
#                     st.warning(f"Unsupported file format: {filename}")
#                     continue
#                 dataframes.append(dataframe)
#                 uploaded_file_names.append(filename)
            
#             # Remove the temporary directory and its contents
#             if os.path.exists(temp_dir):
#                 for root, dirs, files in os.walk(temp_dir, topdown=False):
#                     for name in files:
#                         os.remove(os.path.join(root, name))
#                     for name in dirs:
#                         os.rmdir(os.path.join(root, name))
#                 os.rmdir(temp_dir)
            
#         elif file_extension.lower() in [".csv", ".xls", ".xlsx", "xlsm", ".xlsb", ".txt"]:
#             # Process single file uploads as usual
#             if file_extension.lower() == ".csv":
#                 dataframe = pd.read_csv(uploaded_file)
#             elif file_extension.lower() in [".xls", ".xlsx", "xlsm", ".xlsb"]:
#                 dataframe = pd.read_excel(uploaded_file)
#             elif file_extension.lower() == ".txt":
#                 # Check the second line of the file to determine the delimiter
#                 with uploaded_file as f:
#                     # Skip the first line
#                     next(f)
#                     second_line = f.readline().decode().strip()  # decode bytes to string
#                     delimiter = "|" if "|" in second_line else "\t"  # Assume "|" delimiter for .psv, "\t" for .tsv

#                     # Reset file pointer to the beginning of the file
#                     f.seek(0)

#                     # Read the file using pandas read_csv
#                     dataframe = pd.read_csv(f, delimiter=delimiter)
#             dataframes.append(dataframe)
#             uploaded_file_names.append(uploaded_file.name)
#         else:
#             st.warning(f"Unsupported file format: {file_extension}")

# # Display the dataframes
# if dataframes:
#     for idx, dataframe in enumerate(dataframes):
#         st.write(f"{idx+1}.{uploaded_file_names[idx]}")  # Display file name
#         st.write(dataframe)

# # Allow user to select files for chatting
# if dataframes:
#     selected_file_names = st.multiselect("**Select files for chatting:**", options=uploaded_file_names)

#     # Check if at least one file is selected for chatting
#     if selected_file_names:
#         # Chatbot functionality
#         st.markdown("<h3 style='text-align: center;'>CHAT WITH BOT</h3>", unsafe_allow_html=True)

#         # Define function to generate response from user input
#         def generate_response(input_text, selected_file_names, dataframes):
#             response = ""

#             for file_name in selected_file_names:
#                 # Find the index of the selected file in the uploaded_file_names list
#                 idx = uploaded_file_names.index(file_name)

#                 if "summary" in input_text.lower():
#                     summary_data = round(dataframes[idx].describe(include="all"), 2).T.reset_index()
#                     response += f"**Summary of {file_name}:**\n"
#                     response += summary_data.to_string() + "\n"

#                 elif "missing values" in input_text.lower():
#                     missing_values = dataframes[idx].isnull().sum()
#                     response += f"**Missing values in {file_name}:**\n"
#                     response += missing_values[missing_values > 0].to_string() + "\n"

#                 else:
#                     agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
#                                                           dataframes[idx], verbose=True,
#                                                           agent_type=AgentType.OPENAI_FUNCTIONS,
#                                                           agent_executor_kwargs={"handle_parsing_errors": True})
#                     response += f"**Response for {file_name}:**\n"
#                     response += agent.run(input_text) + "\n"

#             return response

#         # Container for chat history and download button
#         response_container = st.container()
#         download_button_container = st.container()

#         # Container for text box
#         input_container = st.container()

#         with input_container:
#             # Create a form for user input
#             with st.form(key='my_form', clear_on_submit=True):
#                 user_input = st.text_area("**You:**", key='input', height=st.session_state.get('input_height', 50))
#                 st.session_state['input_height'] = len(user_input.split('\n')) * 20  # Adjust height based on input length
#                 submit_button = st.form_submit_button(label='Send')

#             if submit_button and user_input:
#                 # If user submits input, generate response and store input and response in session state variables
#                 try:
#                     response = generate_response(user_input, selected_file_names, dataframes)
#                     st.session_state['past'].append(user_input)
#                     st.session_state['generated'].append(response)
#                 except Exception as e:
#                     st.error("An error occurred: {}".format(e))

#         if st.session_state['generated']:
#             # Display chat history in a container
#             with response_container:
#                 st.markdown("---")  # Adding a border line
#                 for i in range(len(st.session_state['generated'])):
#                     message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
#                     message(st.session_state["generated"][i], key=str(i))

#              # Add a download button for chat history
#             with download_button_container:
#                 download_button_clicked = False  # Flag to track whether download button has been clicked
#                 download_button = st.button("**Download Chat History**")

#                 if download_button and not download_button_clicked:
#                     download_button_clicked = True  # Set flag to indicate button has been clicked
#                     # Save chat history to a Word document
#                     chat_history_file_path = "chat_history.docx"
#                     document = Document()
#                     document.add_heading('Chat History', level=1)
#                     document.add_paragraph()

#                     for user_input, generated_response in zip(st.session_state["past"], st.session_state["generated"]):
#                         user_paragraph = document.add_paragraph()
#                         user_paragraph.add_run("You: ").bold = True
#                         user_paragraph.add_run(user_input).font.size = Pt(12)

#                         bot_paragraph = document.add_paragraph()
#                         bot_paragraph.add_run("Bot: ").bold = True
#                         bot_paragraph.add_run(generated_response).font.size = Pt(12)

#                         # Add some space between user and bot responses
#                         document.add_paragraph()

#                     document.save(chat_history_file_path)

#                     # Provide a download link for the Word document
#                     st.markdown(f"[Download Chat History](sandbox:/path/{chat_history_file_path})", unsafe_allow_html=True)
