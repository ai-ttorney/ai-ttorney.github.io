import fitz 
import json 
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List

#llm = ChatOpenAI(api_key="insert_key",
#               temperature=0.2,  
#                max_tokens=2000
#                )

class Document(BaseModel):
    son: str = Field(description="What is the result of this case? Give the penalty information.")
    seb: str = Field(description="What is the reason for this case? Explain the reason for this cases step by step and be precise.")
    summary: str = Field(description="Post summary")

parser = JsonOutputParser(pydantic_object=Document)

# Function to load non-editable PDF
def load_non_editable_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

pdf_text = load_non_editable_pdf("EmsalKarar/2.pdf") #20-26 arasini almiyor.




prompt = PromptTemplate(
    template=(
        "Extract detailed and precise information according to the specified format. "
        "Provide comprehensive insights, including relevant legal contexts and practical implications. "
        "Avoid generic responses. \n{format_instructions}\n{context}"
    ),
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


chain = prompt | llm | parser

response = chain.invoke({
    "context": pdf_text
})

jsonl_output = {
    "messages": [
        {"role": "system", "content": "AI-ttorney is a financial law advice chatbot and will only answer questions related to Turkey's financial law. For all other questions, it must respond: 'AI-ttorney can't answer the questions outside of Turkey's financial law.'"},
        {"role": "user", "content": response['seb']},
        {"role": "assistant", "content": response['son']}
    ]
}

with open("output.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(jsonl_output, ensure_ascii=False) + "\n")

#print(jsonl_output)
#print(response['seb'])
