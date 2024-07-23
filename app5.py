from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import fitz  # PyMuPDF
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()
port = int(os.environ.get("PORT", 5000))
app.mount("/static", StaticFiles(directory="frontend"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
uploaded_pdf_text = None
class QueryRequest(BaseModel):
    prompt: str
class UploadResponse(BaseModel):
    message: str
    pdf_text: str
class QueryResponse(BaseModel):
    response: str
def get_text_from_pdf(pdf_data):
    pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text
def get_gemini_response(input_text):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_text)
    return response.text
def generate_response(prompt, pdf_text):
    input_text = f"""
    **Retrieval-Augmented Generation (RAG) Response:**
    {pdf_text}
    Generating a detailed and contextually relevant response based on the provided prompt: `{prompt}`.
    Here's how to proceed:
    1. **Retrieve Relevant Information:**
       - Utilize retrieval techniques to gather pertinent data or documents related to `{prompt}`. This could include searching databases, accessing specific repositories, or querying relevant sources.
    2. **Generate Response:**
       - Based on the retrieved information, craft a comprehensive response that addresses the context of `{prompt}`. Ensure the response is coherent, informative, and directly relates to the input.
    3. **Example Structure for Response:**
       - Your generated response should adhere to a structured format, incorporating insights gleaned from the retrieved data. Consider organizing the response to provide clear and actionable information.
    4. **Quality Assurance:**
       - Validate the accuracy and relevance of the generated response against the original query `{prompt}`. Ensure that the response meets expected standards of completeness and depth.
    ---
    **Example:**
    If `{prompt}` refers to a specific query or topic, replace it accordingly to initiate the retrieval and generation process.
    """
    response = get_gemini_response(input_text)
    return response
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.get("/", response_class=HTMLResponse)
# async def get_index(request: Request):
#     return app.get_static_file("index.html")
@app.get("/")
async def read_index_html():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content)

@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.post("/upload")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    global uploaded_pdf_text
    try:
        pdf_data = await pdf_file.read()
        uploaded_pdf_text = get_text_from_pdf(pdf_data)
        return UploadResponse(message="PDF uploaded successfully", pdf_text=uploaded_pdf_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.post("/query", response_model=QueryResponse)
async def query(data: QueryRequest):
    global uploaded_pdf_text
    prompt = data.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt parameter is missing or empty")
    try:
        if uploaded_pdf_text is None:
            raise HTTPException(status_code=400, detail="No PDF text available. Upload a PDF first.")
        # Generate response based on prompt and extracted text
        response = generate_response(prompt, uploaded_pdf_text)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)