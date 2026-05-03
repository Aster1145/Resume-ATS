import spacy
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from the PDF."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def test_custom_ai(pdf_path):
    # 1. LOAD YOUR CUSTOM AI BRAIN
    # Notice we are loading the folder we just created, NOT the generic English model!
    print("Loading Custom Resume AI...")
    nlp = spacy.load("my_custom_resume_model/model-best")
    
    # 2. READ THE PDF
    print(f"Reading {pdf_path}...\n")
    resume_text = extract_text_from_pdf(pdf_path)
    
    if not resume_text:
        return

    # 3. FEED TEXT TO THE AI
    doc = nlp(resume_text)
    
    # 4. PRINT THE RESULTS
    print("=" * 40)
    print("🤖 AI EXTRACTION RESULTS")
    print("=" * 40)
    
    # We will group the found items by their label (Name, College, Skills, etc.)
    extracted_data = {}
    
    for entity in doc.ents:
        label = entity.label_
        text = entity.text.strip().replace('\n', ' ')
        
        if label not in extracted_data:
            extracted_data[label] = set() # Use a set to prevent duplicates
            
        extracted_data[label].add(text)
        
    # Print them out nicely
    for label, items in extracted_data.items():
        print(f"\n📌 {label}:")
        for item in items:
            print(f"  - {item}")

if __name__ == "__main__":
    # Make sure your test.pdf is inside the sample_resumes folder!
    target_resume = "sample_resumes/test.pdf"
    test_custom_ai(target_resume)

"""
TEST EXECUTION RESULTS:
========================================
🤖 AI EXTRACTION RESULTS
========================================

📌 Name:
  - Hayden Lee 123

📌 College Name:
  - Massachusetts Institute of Technology Cambridge, MA Candidate for

📌 Skills:
  - LANGUAGES Mandarin
  - Adobe Photoshop Conceptualized a children’s mediathèque based on field conditions across time Collected real
  - Adobe Photoshop Conceptualized a fitness center to direct viewpoints towards programs of interest Experimented with relationship of carving and packing programs to direct focus towards the space’s center Explored effects of changing wall and ceiling geometries, creating special vantage points in certain locations WORK EXPERIENCE Robotics Company January 20XX Education Design Intern

📌 Designation:
  - Back Bay Children’s Mediathèque February
"""