import sys
sys.dont_write_bytecode = True

import streamlit as st
import numpy as np
import re

def render(document_list: list, meta_data: dict, time_elapsed: float):
  retriever_message = st.expander(f"Response Time & Resume")
  
  with retriever_message:
    st.markdown(f"Total time elapsed: {np.round(time_elapsed, 3)} seconds")

    # Display resume popups
    button_columns = st.columns([0.2, 0.2, 0.2, 0.2, 0.2], gap="small")
    for index, document in enumerate(document_list[:5]):
      with button_columns[index], st.popover(f"Resume {index + 1}"):
        # Remove any shortlisting reason using regex if present
        clean_document = re.split(r'\*\*Shortlisting Reason:\*\*', document)[0].strip()
        st.markdown(clean_document)

def extract_id_from_doc(doc):
  """Extract the ID from a document string"""
  if not isinstance(doc, str):
    return None
    
  import re
  id_match = re.search(r'Applicant ID\s+(\d+)', doc)
  if id_match:
    return id_match.group(1)
  return None

if __name__ == "__main__":
  render(sys.argv[1], sys.argv[2])