import streamlit as st
import openai
import os

# Set your OpenAI API key here (store securely using environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_code_and_suggest(code_snippet):
  """
  Analyzes the code snippet using OpenAI and suggests alternative solutions.

  Args:
    code_snippet (str): The code snippet to be analyzed.

  Returns:
    list: A list of lines containing the suggested alternative solution.
  """
   
  prompt = f"Given the following incomplete code snippet, suggest ways to complete the functionality and provide optimized code:\n\n{code_snippet}\n\nCompletion and Optimization Suggestions:"

  try:
    response = openai.completions.create(
      model="gpt-3.5-turbo-instruct", # Replace with recommended model
      prompt=prompt,
      max_tokens=150, # Limit response length to avoid overwhelming outputs
      n=1, # Request only 1 suggestion
      stop=None, # Don't define a specific stop sequence
      temperature=0.7, # Control creativity (0.7 for balanced suggestions)
    )
     
    # Checking if response is successful
    if response and response.choices and response.choices[0].text:
      # Extracting suggestion from the response
      suggestion = response.choices[0].text.strip()
      suggestion_lines = suggestion.split('\n')
      return suggestion_lines
    else:
      raise ValueError("No response received from OpenAI API.")
  except openai.RateLimitError as a:
    st.error(f"OpenAI API quota exceeded. Please upgrade your plan or consider alternatives. the original error: {a}")
    return []
  except Exception as e:
    # Handling other exceptions
    st.error(f"Error occurred: {e}")
    return []

def main():
  st.title("Code Snippet Completion & Optimization")
   
  # Text area for user input
  code_snippet = st.text_area("Enter your incomplete code snippet here:")
   
  # Button to trigger code analysis
  if st.button("Analyze"):
    suggestion_lines = analyze_code_and_suggest(code_snippet)
     
    # Displaying suggestions
    if suggestion_lines:
      st.subheader("Completion and Optimization Suggestions")
      for line in suggestion_lines:
        st.write(line)
    else:
      st.warning("No suggestions found.")

if __name__ == '__main__':
  main()